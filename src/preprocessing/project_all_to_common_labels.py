from PIL import Image
import os
import sys
sys.path.append(os.path.abspath('..'))
import torch
import numpy as np
from datasets.utils import *
from datasets.class_transformation import *
from torchvision import transforms
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

ego_mask_path = 'ego_mask.png'
gt_transform = transforms.Compose([
    ProjectCarlaToCommonLabels(),
    RGBToID(),
    UpdateMaskBasedOnTrainId(),
    UpdateMaskBasedOnBinaryImage(ego_mask_path, None),
    transforms.Lambda(lambda x: np.array(x, dtype=np.int8))
])

pgt_transform = transforms.Compose([
    ProjectCityScapeToCommonLabels(),
    RGBToID(),
    UpdateMaskBasedOnTrainId(),
    UpdateMaskBasedOnBinaryImage(ego_mask_path, None),
    transforms.Lambda(lambda x: np.array(x, dtype=np.int8))
])

def process_file(img_path):

    if 'pseudos' in img_path:
        transformed_img = pgt_transform(img)
    else:
        return

    save_path = os.path.splitext(img_path)[0]
    if isinstance(transformed_img, torch.Tensor):
        transformed_img = transformed_img.numpy()

    # Save the numpy array
    np.savez_compressed(save_path, transformed_img)


def parallel_apply_transforms(root_dir, num_processes=20):
    # List all .png files in 'semantic_masks' and 'pseudo' folders
    all_files = []
    for dp, dn, filenames in os.walk(root_dir):
        if 'pseudos' in dp:
            all_files.extend([os.path.join(dp, f) for f in filenames if f.endswith('.png')])

    # Limit the number of processes based on available CPU cores and requested processes
    print(len(all_files))
    num_processes = min(num_processes, cpu_count())

    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_file, all_files), total=len(all_files)))


if __name__ == "__main__":
    from arguments import args
    dataset_root = args.dataset
    parallel_apply_transforms(dataset_root)
