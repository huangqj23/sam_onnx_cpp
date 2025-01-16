'''
Author: aliyun9951438140 huangquanjin24@gmail.com
Date: 2023-07-03 17:42:44
LastEditors: aliyun9951438140 huangquanjin24@gmail.com
LastEditTime: 2025-01-16 13:28:32
FilePath: /sam_onnx_cpp/projects/pie-sam/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
from sam import SAM
from utils import HandlingError

def parse_args():
    parser = argparse.ArgumentParser(description='SAM')
    parser.add_argument('--device', default='cpu',
                        type=str, help='cpu or cuda')
    parser.add_argument('--prompts_type', default='json',
                        type=str, help='Type of prompt file.')
    parser.add_argument('--model_type', default='vit_t',  # FSAM Yolov8xFSAM 
                        type=str, help='Which scale of model to use.')
    parser.add_argument('--threads', default=1, type=int,
                        help='Number of threads to use.')
    parser.add_argument('--pre_encode', default=False,
                        type=bool, help='Whether to pre-encode images.')

    arg = parser.parse_args()
    return arg


if __name__ == '__main__':

    args = parse_args()

    if args.model_type == 'vit_t':
        m_type = args.model_type.split('_')[-1]
        path_dict = dict(onnx='./ckpts/onnx/sam_b_quantized.onnx', img_encoder='./ckpts/fsam/mobile_sam.pt')
    else:
        m_type = args.model_type.split('_')[-1]
        path_dict = dict(onnx='./ckpts/onnx/sam_{}_quantized.onnx'.format(m_type), img_encoder='./ckpts/img_encoder/sam_vit_{}.pth'.format(m_type))

    path_dict.update(dict(image='./images', prompts='./prompts/prompts.json', masks='./caches/masks', logits='./caches/logits', image_embeddings='./caches/image_embeddings'))
    try:

        m = SAM
        sam = m(path_dict, prompts_type=args.prompts_type,
                  model_type=args.model_type, device=args.device)
    except Exception as e:
        raise HandlingError('SAM模型初始化错误: ' + repr(e), 101)
    try:
        sam.load()  # 模型加载
    except Exception as e:
        raise HandlingError('SAM模型权重加载错误: ' + repr(e), 102)
    try:
        # 监听prompts.json, 当prompts中的所有记录的status 都为done的时候 程序停止退出
        sam.forward(pre_encode=args.pre_encode,
                    multi_process=args.threads > 1, thread=args.threads)
    except Exception as e:
        print(e.handling_msg)
    del sam
