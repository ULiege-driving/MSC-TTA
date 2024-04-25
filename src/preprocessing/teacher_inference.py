import os
import numpy as np
import cv2
import torch
from tqdm import tqdm
from glob import glob
from natsort import natsorted
import inspect

from mmseg.apis import inference_model, init_model, show_result_pyplot, MMSegInferencer

def segment(data_path, model_name, args):

    pseudo_gt_path = os.path.join(data_path, 'pseudo')
    img_path = os.path.join(data_path, 'images')

    os.makedirs(pseudo_gt_path, exist_ok=True)

    img_tree = os.listdir(img_path)
    for folder in img_tree:
        os.makedirs(os.path.join(pseudo_gt_path, folder), exist_ok=True)

    model = build_model(model_name)

    for folder in img_tree:
        img_list = os.listdir(os.path.join(img_path, folder))
        img_list.sort()
        for img in tqdm(img_list):
            write_path = os.path.join(pseudo_gt_path, folder, os.path.splitext(img)[0])
            frame = cv2.imread(os.path.join(img_path, folder, img))  # check channel order !!

            #out = MMSegInferencer(model_name, )
            out = inference_model(model, frame)

            #np.save(write_path, out.pred_sem_seg.data.cpu().numpy())
            show_result_pyplot(model, frame, out, show=False, save_dir=os.path.join(pseudo_gt_path, folder), out_file=write_path+".png", opacity=1)


def build_model(model_name):
    if model_name == "pspnet":
        config = os.path.join('../code', '..', 'mmsegmentation', 'configs', 'pspnet',
                              'pspnet_r101b-d8_512x1024_80k_cityscapes.py')
        checkpoint = os.path.join('../code', '..', 'mmsegmentation', 'checkpoints',
                                  'pspnet_r101b-d8_512x1024_80k_cityscapes_20201226_170012-3a4d38ab.pth')
    elif model_name == "pointrend_seg":
        config = os.path.join('../code', 'mmsegmentation', 'configs', 'point_rend',
                              'pointrend_r101_512x1024_80k_cityscapes.py')
        checkpoint = os.path.join('../code', '..', 'mmsegmentation', 'checkpoints',
                                  'pointrend_r101_512x1024_80k_cityscapes_20200711_170850-d0ca84be.pth')
    elif model_name == "segformer":
        config = os.path.join('../code', '..', 'mmsegmentation', 'configs', 'segformer',
                              'segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py')
        checkpoint = os.path.join('../code', '..', 'mmsegmentation', 'checkpoints',
                                  'segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth')
    elif model_name == "setr":
        config = os.path.join('../code', '..', 'mmsegmentation', 'configs', 'setr',
                              'setr_vit-l_pup_8xb1-80k_cityscapes-768x768.py')
        checkpoint = os.path.join('../code', '..', 'mmsegmentation', 'checkpoints',
                                  'setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth')
    return init_model(config, checkpoint, device='cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    from arguments import args
    for dts in args.dataset:
        for dir in os.listdir(dts):
            segment(os.path.join(dts, dir), args.teacher, args)
