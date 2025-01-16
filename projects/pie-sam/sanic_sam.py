# Todo: Using sanic to improve the speed of prediction.
import os
import torch
import numpy as np
import cv2
import time
from concurrent import futures
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from utils import json2prompts, arrays2prompts, txt2prompts, load_img, fuse_logits, HandlingError, load_logits, coords_b2s


class SAM:
    def __init__(self,
                 path: dict,
                 prompts_type='json',
                 model_type='vit_h',
                 device='cuda',

                 ):
        self.path = path  # 包含模型存储路径、正负点缓存路径、每次预测结果缓存路径、最终Mask存储路径
        assert prompts_type in (
            'arrays', 'json', 'txt'), 'Not supported input type !'  # 仅支持三种prompts的输入类别
        self.prompts_type = prompts_type
        self.model_type = model_type
        self.device = device
        # 初始化存储模型变量
        self.model = None
        self.session = None
        self.image_encoder = None
        self.predictor = None

    # 加载onnx_[prompt_encoder | mask_decoder] 和 torch_image_encoder.predictor 到类方法
    def load(self):
        onnx_path, img_encoder_path = self.path['onnx'], self.path['img_encoder']
        self.session = onnxruntime.InferenceSession(onnx_path,
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        sam = sam_model_registry[self.model_type](checkpoint=img_encoder_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    # 整合onnx的输入
    def prep(self, image_embeddings, img_size, points, labels, logits):

        points = np.concatenate([points, np.array([[0.0, 0.0]])], axis=0)[
            None, :, :]
        labels = np.concatenate(
            [labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
        points = self.predictor.transform.apply_coords(
            points, img_size).astype(np.float32)
        sess_input = {
            "image_embeddings": image_embeddings,
            "point_coords": points,
            "point_labels": labels,
            "mask_input": logits,
            "has_mask_input": np.zeros(1, dtype=np.float32) if np.sum(logits) == 0 else np.ones(1, dtype=np.float32),
            "orig_im_size": np.array(img_size, dtype=np.float32)
        }
        return sess_input

    # 预测
    def pred(self, x):
        masks, _, logits = self.session.run(None, x)
        masks = masks > self.predictor.model.mask_threshold
        return masks, logits

    # 后处理，最终结果保存前的处理
    @staticmethod
    def post(masks):
        masks = masks.squeeze(0).transpose(1, 2, 0)
        return masks

    # 将输入prompts转换为onnx要求的格式
    def get_prompts(self, inputs, win_id):
        if self.prompts_type == 'arrays':  # 假设格式已经正确
            prompts = arrays2prompts(inputs)
        elif self.prompts_type == 'json':
            prompts = json2prompts(inputs, win_id)
        else:
            prompts = txt2prompts(inputs)

        return prompts

    # 缓存结果 mask和low_res_logits到文件
    def put_result(self, masks, logits, cls_idx, win_idx, c):
        cv2.imwrite(os.path.join(self.path['masks'], 'mask_{}_{}_{}.jpg'.format(str(cls_idx), str(win_idx), str(c))),
                    self.post(masks).astype(np.uint8) * 255)
        np.save(os.path.join(self.path['logits'], 'logits_{}_{}_{}.npy'.
                             format(str(cls_idx), str(win_idx), str(c))), logits)

    def fusion(self, logits, center, cls_id, img_size):
        # 融合logits to x_y_h_w_fusion_logits.npy
        fuse_logits(logits, self.path['logits'], center, cls_id, img_size)

    # 完整的窗口推测流程
    def forward_window(self, cls_id=-1, win_id=-1, logits=None, pre_encode=False, infos=None, center=None, img_size=None, image_embeddings=None):

        # 循环根据点位对结果进行更新
        c = 0

        # 重复读取prompts缓存文件，传回None时 表示当前窗口处理完成
        # 反之一直查询prompts缓存，并根据缓存更新输入，迭代预测
        m_time_last = 0
        end = False
        while True:
            # 判断文件是否有更改，有更改才执行下一步
            m_time = os.path.getmtime(self.path['prompts'])
            if m_time <= m_time_last:
                continue
            m_time_last = m_time
            try:
                prompts = self.get_prompts(
                    self.path['prompts'], win_id)
            except Exception as e:
                raise HandlingError('Prompts读取错误: ' + str(e), 105)

            if prompts is None:
                end = True
                break
            elif prompts[0] is None and isinstance(prompts[-1], int):
                # 缓存融合
                if prompts[-2] == cls_id:
                    print('\nlogits 缓存融合中...')
                    try:
                        self.fusion(logits, center, cls_id, img_size)
                    except Exception as e:
                        raise HandlingError('Logits缓存融合错误: ' + str(e), 109)
                    print('logits 缓存融合完成...\n')

                # 切换窗口
                win_id = prompts[-1]
                cls_id = prompts[-2]
                print("切换窗口 id: {}...".format(str(win_id)))
                image, center = load_img(self.path['image'], win_id)
                img_size = image.shape[:2]
                print('图像编码中...')
                image_embeddings = self.img_embed(image)
                print('图像编码完成')
                break
            else:

                if prompts[-2] != cls_id:
                    c = 0
                    logits = None
                    print("切换类别 id: {}...".format(str(prompts[-2])))
                points, labels, cls_id, current_win_id = prompts
                try:
                    if pre_encode:
                        win_id = current_win_id
                        image_embeddings = np.load(os.path.join(
                            self.path['image_embeddings'], 'image_embeddings_{}.npy'.format(str(win_id))))
                        center, img_size = infos[win_id]
                    else:
                        print('类别：{} 窗口：{} ...'.format(
                            cls_id, current_win_id))

                        if win_id == -1:
                            image, center = load_img(
                                self.path['image'], current_win_id)
                            img_size = image.shape[:2]
                            print('图像编码中...')
                            image_embeddings = self.img_embed(image)
                            print('图像编码完成')
                        win_id = current_win_id
                except Exception as e:
                    raise HandlingError('图像加载错误: ' + str(e), 103)

                try:
                    pts = []
                    print(center, img_size)
                    for coords in points:
                        pts.append(coords_b2s(coords, center, img_size))
                    points, labels = np.array(pts), np.array(labels)
                    logits = np.zeros((1, 1, 256, 256), dtype=np.float32) if logits is None else \
                        load_logits(self.path['logits'], logits, center, cls_id, c, img_size).astype(
                            np.float32)
                except Exception as e:
                    raise HandlingError('logits加载错误: ' + str(e), 104)
                try:
                    sess_input = self.prep(
                        image_embeddings, img_size, points, labels, logits=logits)
                except Exception as e:
                    raise HandlingError('数据预处理错误: ' + str(e), 106)
                try:
                    masks, logits = self.pred(sess_input)
                except Exception as e:
                    raise HandlingError('模型推断错误: ' + str(e), 107)
                try:
                    self.put_result(masks, logits, cls_id, win_id, c)
                except Exception as e:
                    raise HandlingError('结果写入错误: ' + str(e), 108)
            print("{}号类 {}号窗口 第{}次标注完成...\n".format(
                str(cls_id), str(win_id), str(c + 1)))
            c += 1

        return end, cls_id, win_id, logits, center, img_size, image_embeddings

    def img_embed(self, image):
        # 图像编码
        try:
            with torch.no_grad():
                self.predictor.set_image(image)
                image_embeddings = self.predictor.get_image_embedding().cpu().numpy()
        except Exception as e:
            raise HandlingError('图像编码错误: ' + str(e), 104)
        return image_embeddings

    def forward(self, pre_encode=False, multi_process=False, cls_id=-1, win_id=-1, thread=1):

        infos = self.all_img_encode(
            multi_process=multi_process, thread=thread) if pre_encode else None
        logits = None
        print('开始标注...')
        center, img_size, image_embeddings = None, None, None
        while True:

            end, cls_id, win_id, logits, center, img_size, image_embeddings = self.forward_window(
                cls_id, win_id, logits, pre_encode=pre_encode, infos=infos, center=center, img_size=img_size, image_embeddings=image_embeddings)

            if end:
                print('\n全部窗口标注完成, 程序退出...')
                break

    def img_pre_embed(self, win_id):
        # 图像编码
        try:
            image, center = load_img(self.path['image'], win_id)
            img_size = image.shape[:2]
            with torch.no_grad():
                self.predictor.set_image(image)
                image_embeddings = self.predictor.get_image_embedding().cpu().numpy()
                np.save(os.path.join(self.path['image_embeddings'], 'image_embeddings_{}.npy'.format(str(win_id))),
                        image_embeddings)
        except Exception as e:
            raise HandlingError('图像编码错误: ' + str(e), 104)

        return center, img_size

    # 所有图片预编码
    def all_img_encode(self, multi_process=False, thread=16):
        # 计算窗口数量
        win_num = len([x for x in os.listdir(self.path['image'])
                      if os.path.isfile(os.path.join(self.path['image'], x))])
        # 打印窗口数量，是否使用多线程，线程数
        print('开始预编码...\n')
        t0 = time.perf_counter()
        on_off = '开启' if multi_process else '关闭'
        print('窗口数量: {}, 多线程: {}, 线程数: {}'.format(win_num, on_off, thread))
        if not multi_process:
            infos = []
            for win_id in range(win_num):
                center, img_size = self.img_pre_embed(win_id)
                infos.append((center, img_size))
            t1 = time.perf_counter()
            print('预编码完成, 耗时: {:.2f}s\n'.format(t1 - t0))
            return infos
        else:
            with futures.ThreadPoolExecutor(thread) as executor:
                infos = {executor.submit(self.img_pre_embed, win_id)
                         for win_id in range(win_num)}
            infos = [x.result() for x in infos]
            t1 = time.perf_counter()
            print('预编码完成, 耗时: {:.2f}s\n'.format(t1 - t0))
            return infos


import json
import os
import numpy as np
import cv2


class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg


# 读取文件或者数组转换成输入格式
def json2prompts(inputs, win_id):
    with open(inputs) as f:
        datas = json.load(f)

    current_data = datas[-1]  # 最后一个cls

    data = [x for x in current_data['prompts']
            if x['status'] == 'processing']  # 每次扫描读取正在处理窗口的数据
    if len(data) == 0:
        return None
    data = data[0]

    if data['win_id'] != win_id and win_id != -1:
        return None, None, None, current_data['cls_id'], data['win_id']

    return data['points'], data['labels'], current_data['cls_id'], data['win_id']


def arrays2prompts(inputs):
    return inputs


def txt2prompts(inputs):
    return inputs


def load_img(img_path, win_idx):  # 找到文件夹下对应processing_id的图片
    imgs = [x for x in os.listdir(img_path) if os.path.isfile(
        os.path.join(img_path, x))]
    img = [x for x in imgs if int(
        x.split('.')[0].split('_')[-1]) == win_idx][0]
    center = [int(x) for x in img.split('_')[:2]]
    img = cv2.imread(os.path.join(img_path, img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, center


def coords_b2s(tif_coords, center_coords, img_size):  # 将大图坐标转换为模型使用的小图坐标
    h, w = img_size
    x = tif_coords[0] - center_coords[0] + w // 2
    y = tif_coords[1] - center_coords[1] + h // 2
    return x, y


def coords_s2b(small_coords, center_coords, h, w):  # 将小图坐标转换为大图坐标
    x = small_coords[0] + center_coords[0] - w // 2
    y = small_coords[1] + center_coords[1] - h // 2
    return x, y


def load_logits(path, logits, center, cls_id, c, img_size):

    old_fusion_name = None
    for npy in os.listdir(path):
        if npy.endswith('fusion_logits_{}.npy'.format(str(cls_id))):
            old_fusion_name = npy
            break
    if old_fusion_name is None or c > 0:  # nonzero 但是又没有fusion文件-> 第一个窗口还未标注完成的情况
        # 或者是窗口除第一次外的标注
        return logits
    old_fusion_path = os.path.join(path, old_fusion_name)
    # 每次logits进行融合时候以融合后的中心点坐标和长宽命名: x_y_h_w_fusion_logits.npy
    x, y, h, w = [int(x) for x in old_fusion_name.split('_')[:4]]
    # 转换到顶点方便求重叠区域 和外接矩形
    x1, y1, x2, y2 = xyhw2xyxy(x, y, h, w)
    # 新建的窗口如果和旧的fusion_logits重叠的中心点落点范围
    start_x, end_x = x1 - img_size[0] // 2, x2 + img_size[0] // 2
    # 不用做 < 0的判断，假设在软件已经对可操作区域做padding.
    start_y, end_y = y1 - img_size[1] // 2, y2 + img_size[1] // 2
    # 若新的窗口与旧的fusion_logits坐标重叠
    if (start_x < center[0] < end_x) and (start_y < center[1] < end_y):
        # 以两个框的外接矩形的中心点新建坐标系，并将框转换到其下
        bbox1, bbox2, h_new, w_new, _ = convert_xy(
            x, y, h, w, center, img_size)
        zero = np.zeros((1, 1, h_new, w_new))  # 新建全零张量
        zero[:, :, bbox1[1]: bbox1[3], bbox1[0]: bbox1[2]] = np.load(
            old_fusion_path)  # 给对应旧fusion_logits位置赋值
        # 裁取新建窗口位置作为logits
        return zero[:, :, bbox2[1]: bbox2[3], bbox2[0]: bbox2[2]]
    else:
        return np.zeros((1, 1, 256, 256), dtype=np.float32)


def convert_xy(x, y, h, w, center, img_size):
    x1, y1, x2, y2 = xyhw2xyxy(x, y, h, w)
    x3, y3, x4, y4 = xyhw2xyxy(center[0], center[1], *img_size)  # 新建窗口的顶点坐标
    new_box = min(x1, x3), min((y1, y3)), max(x2, x4), max(y2, y4)  # 外接矩形框坐标
    # 外接矩形框中心点坐标(大图)
    center_new = (new_box[0] + new_box[2]) // 2, (new_box[1] + new_box[3]) // 2
    h_new = new_box[3] - new_box[1]  # 外接矩形h
    w_new = new_box[2] - new_box[0]  # 外接矩形w
    bbox1 = coords_b2s((x, y), center_new, (h_new, w_new))  # 转换坐标
    bbox2 = coords_b2s(center, center_new, (h_new, w_new))  # 转换坐标
    bbox1 = xyhw2xyxy(bbox1[0], bbox1[1], h, w)
    bbox2 = xyhw2xyxy(bbox2[0], bbox2[1], *img_size)
    # 转到low_res坐标系
    scale = img_size[0] // 256
    # img_size resize_to 1024 -> encode_to 256
    h_new, w_new = h_new // scale, w_new // scale
    bbox1 = [x // scale for x in bbox1]
    bbox2 = [x // scale for x in bbox2]
    return bbox1, bbox2, h_new, w_new, center_new


def fuse_logits(logits, path, center, cls_id, img_size):
    old_fusion_name = None
    for npy in os.listdir(path):
        if npy.endswith('fusion_logits_{}.npy'.format(str(cls_id))):
            old_fusion_name = npy
            break
    if old_fusion_name is None:
        np.save(os.path.join(path, '{}_{}_{}_{}_fusion_logits_{}.npy'.
                             format(str(center[0]), str(center[1]), str(img_size[1]), str(img_size[0]), str(cls_id))), logits)
    else:
        old_fusion = np.load(os.path.join(path, old_fusion_name))
        x, y, h, w = [int(x) for x in old_fusion_name.split('_')[:4]]
        bbox1, bbox2, h_new, w_new, center_new = convert_xy(
            x, y, h, w, center, img_size)
        zero = np.zeros((1, 1, h_new, w_new))  # 新建全零张量
        # print(zero.shape, old_fusion.shape)
        zero[:, :, bbox1[1]: bbox1[3], bbox1[0]: bbox1[2]] = old_fusion
        zero[:, :, bbox2[1]: bbox2[3], bbox2[0]: bbox2[2]] = logits
        scale = img_size[0] // 256
        H, W = h_new * scale, w_new * scale
        os.remove(os.path.join(path, old_fusion_name))
        np.save(os.path.join(path, '{}_{}_{}_{}_fusion_logits_{}.npy'
                             .format(str(center_new[0]), str(center_new[1]), str(H), str(W), str(cls_id))), zero)


def xyhw2xyxy(x, y, h, w):
    x1 = x - w // 2
    x2 = x1 + w
    y1 = y - h // 2
    y2 = y1 + h
    return x1, y1, x2, y2
