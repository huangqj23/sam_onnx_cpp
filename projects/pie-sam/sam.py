import os
import torch
import numpy as np
import cv2
import time
from concurrent import futures

import onnxruntime
from utils import json2prompts, arrays2prompts, txt2prompts, load_img, fuse_logits, HandlingError,\
    load_logits, coords_b2s


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
        img_encoder_path = self.path['img_encoder']
        if self.model_type != 'vit_t':
            from segment_anything import sam_model_registry, SamPredictor
            onnx_path = self.path['onnx']
            self.session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            sam = sam_model_registry[self.model_type](checkpoint=img_encoder_path)            
        else:
            from mobile_sam import sam_model_registry, SamPredictor
            sam = sam_model_registry[self.model_type](checkpoint=img_encoder_path)  
            

        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    # 整合onnx的输入
    def prep(self, image_embeddings, img_size, points, labels, logits):

        if self.model_type != 'vit_t':
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
        else:
            sess_input = points, labels, None, logits.squeeze(0), False, True
        return sess_input

    # 预测
    def pred(self, x):
        if self.model_type != 'vit_t':
            masks, _, logits = self.session.run(None, x)
            masks = masks > self.predictor.model.mask_threshold
        else:
            masks, _, logits = self.predictor.predict(*x)
        return masks, logits

    # 后处理，最终结果保存前的处理
    
    def post(self, masks):
        if self.model_type != 'vit_t':
            masks = masks.squeeze(0).transpose(1, 2, 0)
        else:
            masks = masks.transpose(1, 2, 0)
        return masks

    # 将输入prompts转换为onnx要求的格式
    def get_prompts(self, inputs):
        if self.prompts_type == 'arrays':  # 假设格式已经正确
            prompts = arrays2prompts(inputs)
        elif self.prompts_type == 'json':
            prompts = json2prompts(inputs)
        else:
            prompts = txt2prompts(inputs)

        return prompts

    # 缓存结果 mask和low_res_logits到文件
    def put_result(self, masks, logits=None, cls_id=-1, win_id=-1, c=0):
        cv2.imwrite(os.path.join(self.path['masks'], 'mask_{}_{}_{}.jpg'.format(str(cls_id), str(win_id), str(c))),
                    self.post(masks).astype(np.uint8) * 255)
        if logits is not None:
            if self.model_type == 'vit_t':
                logits = logits[None, :, :, :]
            np.save(os.path.join(self.path['logits'], 'logits_{}_{}_{}.npy'.
                                 format(str(cls_id), str(win_id), str(c))), logits)

    def fusion(self, logits, center, cls_id, img_size):
        # 融合logits to x_y_h_w_fusion_logits.npy
        fuse_logits(logits, self.path['logits'], center, cls_id, img_size)

    # 完整的窗口推测流程
    def forward_window(self, pre_encode=False, infos=None, fast=False):
        cls_id = -1
        win_id = -1
        logits = None
        center = None
        img_size = None
        image_embeddings = None
        # 循环根据点位对结果进行更新
        c = 0

        # 重复读取prompts缓存文件，传回None时 表示当前窗口处理完成
        # 反之一直查询prompts缓存，并根据缓存更新输入，迭代预测
        m_time_last = 0
        while True:
            # 判断文件是否有更改，有更改才执行下一步
            m_time = os.path.getmtime(self.path['prompts'])
            if m_time <= m_time_last:
                continue
            m_time_last = m_time
            # 读取修改后的文件
            try:
                prompts = self.get_prompts(
                    self.path['prompts'])
            except Exception as e:
                raise HandlingError('Prompts读取错误: ' + str(e), 105)
            # 循环结束条件
            if prompts is None:
                print('\n全部窗口标注完成，程序退出...')
                break
            elif prompts == 'continue':
                continue
            else:
                # 一旦类别变化就将count和logits初始化
                if prompts[-2] != cls_id and cls_id != -1:
                    c = 0
                    logits = None
                    print("切换类别 id: {}...".format(str(prompts[-2])))
                try:
                    # 仅当第一次和查询到win_id变化时才重新编码
                    if win_id == -1 or prompts[-1] != win_id:
                        if win_id == -1:
                            print('开始标注...')
                        else:
                            print("切换窗口 {} to {}...".format(
                                str(win_id), str(prompts[-1])))
                            # 缓存融合
                            if prompts[-2] == cls_id and not fast:
                                print('\nlogits 缓存融合中...')
                                try:
                                    if self.model_type == 'vit_t':
                                        logits = logits[None, :, :, :]
                                    self.fusion(logits, center,
                                                prompts[-2], img_size)
                                except Exception as e:
                                    raise HandlingError(
                                        'Logits缓存融合错误: ' + str(e), 109)
                                print('logits 缓存融合完成...\n')
                        if pre_encode:
                            print('加载预编码...')
                            image_embeddings = np.load(os.path.join(
                                self.path['image_embeddings'], 'image_embeddings_{}.npy'.format(str(prompts[-1]))))
                            center, img_size = infos[prompts[-1]]
                            print('预编码加载完成')
                        else:
                            if fast:
                                img_path, img_size, center = load_img(self.path['image'], prompts[-1], fast=True)
                                print('图像编码中...')
                                image_embeddings = self.img_embed(img_path)
                                print('图像编码完成')
                            else:
                                image, center = load_img(
                                self.path['image'], prompts[-1], fast=fast)
                                img_size = image.shape[:2]
                                print('图像编码中...')
                                image_embeddings = self.img_embed(image)
                                print('图像编码完成')
                except Exception as e:
                    raise HandlingError('图像加载错误: ' + str(e), 103)
                # 获取prompts
                points, labels = prompts[:2]
                try:
                    # 处理prompts
                    pts = []
                    for coords in points:
                        pts.append(coords_b2s(coords, center, img_size))
                    
                    if not fast:
                        points, labels = np.array(pts), np.array(labels)
                        zero = np.zeros((1, 1, 256, 256), dtype=np.float32) 
                        logits = zero if logits is None else \
                        load_logits(self.path['logits'], logits, center, cls_id, c, img_size).astype(
                            np.float32)
                        if len(logits.shape) == 3:
                            logits = logits[None, :, :, :]
                    else:
                        points = pts
                except Exception as e:
                    raise HandlingError('prompts加载错误: ' + str(e), 104)
                try:
                    if not fast:
                        sess_input = self.prep(
                            image_embeddings, img_size, points, labels, logits=logits)
                    else:
                        sess_input = (points, labels), image_embeddings

                except Exception as e:
                    raise HandlingError('数据预处理错误: ' + str(e), 106)
                
                if fast:
                    masks = self.pred(*sess_input)
                else:
                    masks, logits = self.pred(sess_input)

                # 在读取到的win_id变化时，如果cls_id未变化则进行缓存融合
                # 更新ids
                cls_id, win_id = prompts[2:]

                try:
                    self.put_result(masks, logits, cls_id, win_id, c)
                except Exception as e:
                    raise HandlingError('结果写入错误: ' + str(e), 108)

                print("{}号类 {}号窗口 第{}次标注完成...\n".format(
                    str(cls_id), str(win_id), str(c + 1)))
                c += 1

    def img_embed(self, image):
        # 图像编码
        try:
            with torch.no_grad():
                self.predictor.set_image(image)
                image_embeddings = self.predictor.get_image_embedding().cpu().numpy()
        except Exception as e:
            raise HandlingError('图像编码错误: ' + str(e), 104)
        return image_embeddings

    def forward(self, pre_encode=False, multi_process=False, thread=1):

        infos = self.all_img_encode(
            multi_process=multi_process, thread=thread) if pre_encode else None
        self.forward_window(pre_encode=pre_encode, infos=infos)

    def img_pre_embed(self, win_id):
        # 图像编码
        try:
            image, center = load_img(self.path['image'], win_id)
            img_size = image.shape[:2]
            image_embeddings = self.img_embed(image)
            np.save(os.path.join(self.path['image_embeddings'], 'image_embeddings_{}.npy'.format(str(win_id))),
                    image_embeddings)
        except Exception as e:
            raise HandlingError('图像编码错误: ' + str(e), 104)

        return center, img_size, win_id

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
                center, img_size, _ = self.img_pre_embed(win_id)
                infos.append((center, img_size))
            t1 = time.perf_counter()
            print('预编码完成, 耗时: {:.2f}s\n'.format(t1 - t0))
            return infos
        else:
        
            with futures.ThreadPoolExecutor(thread) as executor:
                infos = executor.map(self.img_pre_embed, range(win_num))
            infos = [x for x in infos]  # 多线程顺序不一致
            # 根据win_id排序
            infos.sort(key=lambda x: x[2])
            infos = [(x[0], x[1]) for x in infos]

            t1 = time.perf_counter()
            print('预编码完成, 耗时: {:.2f}s\n'.format(t1 - t0))
            return infos
