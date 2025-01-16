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
def json2prompts(inputs):
    with open(inputs) as f:
        datas = json.load(f)
    if datas[0].get('initialize') == 1:
        return 'continue'
    current_data = datas[-1]  # 最后一个cls

    data = [x for x in current_data['prompts']
            if x['status'] == 'processing']  # 每次扫描读取正在处理窗口的数据
    if len(data) == 0:
        return None
    data = data[0]

    return data['points'], data['labels'], current_data['cls_id'], data['win_id']


def arrays2prompts(inputs):
    return inputs


def txt2prompts(inputs):
    return inputs


def load_img(img_path, win_id, fast=False):  # 找到文件夹下对应processing_id的图片
    imgs = [x for x in os.listdir(img_path) if os.path.isfile(
        os.path.join(img_path, x))]
    img = [x for x in imgs if int(
        x.split('.')[0].split('_')[-1]) == win_id][0]
    center = [int(x) for x in img.split('_')[:2]]
    im = cv2.imread(os.path.join(img_path, img))
    if fast:
        return img_path + '/' + img,  im.shape[:2], center
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im, center


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
        np.save(os.path.join(path, '{}_{}_{}_{}_fusion_logits_{}.npy'
                             .format(str(center[0]), str(center[1]),
                                     str(img_size[1]), str(img_size[0]), str(cls_id))), logits)
    else:
        old_fusion = np.load(os.path.join(
            path, old_fusion_name))
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
