from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from abc import abstractmethod
import numpy as np
import torch
import cv2
from PIL import Image
from datasets.utils import *
from datasets.class_transformation import *
from datasets.classes import *
import random
from tqdm import tqdm
import copy


class OfflineDataset:

    def __init__(self, args, env_id, start_time, end_time):
        # Save the arguments
        self.args = args
        self.num_classes = args.num_classes
        self.env_id = env_id

        self.root_folder = args.dataset

        args.student_subset = 1

        if args.cell_type == 'common':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(-1, -1, -1),
                args=args,
                cached=True)
        elif args.cell_type == 'specific':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=self.env_id,
                args=args,
                cached=True)
        elif args.cell_type == 'spatial':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(self.env_id[0], -1, -1),
                args=args,
                cached=True)
        elif args.cell_type == 'weather':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(-1, self.env_id[1], -1),
                args=args,
                cached=True)
        elif args.cell_type == 'daylight':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(-1, -1, self.env_id[2]),
                args=args,
                cached=True)
        elif args.cell_type == 'weatherdaylight':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(-1, self.env_id[1], self.env_id[2]),
                args=args,
                cached=True)
        elif args.cell_type == 'standalone':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_vehicle_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                vehicle_folder=self.env_id,
                args=args,
                cached=True)

        indices = list(range(len(self.image_paths)))
        random.shuffle(indices)
        self.train_indices = np.array(indices[int(args.validation_fraction * len(indices)):]).astype(int)
        self.val_indices = np.array(indices[:int(args.validation_fraction * len(indices))]).astype(int)


    def get_train_val(self):
        # train set
        train_indices = self.train_indices
        val_indices = self.val_indices
        if len(train_indices) <= 0:
            train_set = None
        else:
            train_set = TorchDataset(np.array(self.image_paths)[train_indices], np.array(self.pseudo_paths)[train_indices],
                                     np.array(self.semantic_mask_paths)[train_indices], np.array(self.timestamps)[train_indices], self.args,
                                     test=False)

        # val set
        if len(val_indices) <= 0:
            val_set = None
        else:
            val_set = TorchDataset(np.array(self.image_paths)[val_indices], np.array(self.pseudo_paths)[val_indices],
                                   np.array(self.semantic_mask_paths)[val_indices], np.array(self.timestamps)[val_indices],
                                   self.args, test=True)
        return DataLoader(train_set, batch_size=self.args.batchsize, shuffle=True, num_workers=self.args.numworkers,
                          pin_memory=False) if train_set is not None else None, DataLoader(val_set, batch_size=self.args.batchsize, shuffle=False,
                                                        num_workers=self.args.numworkers,
                                                        pin_memory=False) if val_set is not None else None


class OnlineDataset:

    def __init__(self, args, env_id, start_time, end_time):
        # Save the arguments
        self.args = args
        self.num_classes = args.num_classes
        self.env_id = env_id
        self.train_step = args.training_step

        # Choose between train or test mode
        self.train_mode = True

        self.root_folder = args.dataset

        # Grab all data in this environment if scenario is 'specific' or 'no_env':
        if args.cell_type == 'common':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(-1, -1, -1),
                args=args,
                cached=True)
        elif args.cell_type == 'specific':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=self.env_id,
                args=args,
                cached=True)
        elif args.cell_type == 'spatial':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(self.env_id[0], -1, -1),
                args=args,
                cached=True)
        elif args.cell_type == 'weather':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(-1, self.env_id[1], -1),
                args=args,
                cached=True)
        elif args.cell_type == 'daylight':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(-1, -1, self.env_id[2]),
                args=args,
                cached=True)
        elif args.cell_type == 'weatherdaylight':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=(-1, self.env_id[1], self.env_id[2]),
                args=args,
                cached=True)
        elif args.cell_type == 'standalone':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_vehicle_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                vehicle_folder=self.env_id,
                args=args,
                cached=True)
        self.first_timestamp = min(self.timestamps) if self.timestamps else None
        self.last_timestamp = max(self.timestamps) if self.timestamps else None
        self.current_position = self.first_timestamp - 1 if self.timestamps else None
        self.BufferSize = args.buffer_size

    def __len__(self):
        return len(self.image_paths)

    def create_testset(self, start, stop):
        imgs, pgts, gts, tss = get_subsets_period(start,
                                                  stop,
                                                  self.image_paths,
                                                  self.pseudo_paths,
                                                  self.semantic_mask_paths,
                                                  self.timestamps)

        dataset = TorchDataset(imgs, pgts, gts, tss, self.args, test=True)
        if len(imgs) < 1 or len(pgts) < 1 or len(gts) < 1 or len(tss) < 1:
            return None
        if len(imgs) != len(pgts) != len(gts) != len(tss):
            return None

        return (DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers,
                           pin_memory=False))

    def create_custom_set(self, start, length):
        imgs, pgts, gts, tss = get_subsets_length(start,
                                                  length,
                                                  self.image_paths,
                                                  self.pseudo_paths,
                                                  self.semantic_mask_paths,
                                                  self.timestamps)

        if len(imgs) <= 0:
            return None
        dataset = TorchDataset(imgs, pgts, gts, tss, self.args, test=True)
        return (DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers,
                           pin_memory=False))

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def create_trainset(self):
        pass


def read_frame(video_path, position):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, position)
    _, frame = video.read()
    video.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class TorchDataset(Dataset):

    def __init__(self, imgs, pgts, gts, tss, args, test=False):
        self.image_paths = imgs
        self.pseudo_paths = pgts
        self.semantic_mask_paths = gts
        self.timestamps = tss
        self.args = args
        self.word = 'arr_0'
        self.img_transform = transforms.Compose([
            #transforms.Resize(COMMON_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self.gt_transform = transforms.Compose([
            ProjectCommonLabelsToCItyscapes20(),
            transforms.Lambda(lambda x: torch.from_numpy(x).long()),
        ])
        self.pgt_transform = transforms.Compose([
            ProjectCommonLabelsToCItyscapes20(),
            transforms.Lambda(lambda x: torch.from_numpy(x).long()),
        ])




        self.use_gt = False  # self.args.groundtruth if test else self.args.teacher

    def __getitem__(self, index):
        # Load image and apply transformation
        img_path = self.image_paths[index] #[0]
        image = Image.open(img_path)
        image = self.img_transform(image)[:3].squeeze()

        # Load pseudo labels and apply transformation
        pgt_path = self.pseudo_paths[index] #[0]
        pseudo = np.load(pgt_path)  # need pseudo labels
        pseudo = self.pgt_transform(pseudo[self.word]).squeeze()

        # Load semantic masks and apply transformation
        gt_path = self.semantic_mask_paths[index] #[0]
        mask = np.load(gt_path)
        mask = self.gt_transform(mask[self.word]).squeeze()

        # Load timestamp
        timestamp = self.timestamps[index]  # [0]

        # Return sample as a tuple
        # print(image.shape, pseudo.shape, mask.shape, timestamp, img_path)
        return image, pseudo, mask, timestamp, img_path

    def __len__(self):
        return len(self.image_paths)

class CachedOnlineDataset:

    def __init__(self, args, env_id, start_time, end_time):
        # Save the arguments
        self.args = args
        self.num_classes = args.num_classes
        self.env_id = env_id
        self.train_step = args.training_step

        # Choose between train or test mode
        self.train_mode = True

        self.root_folder = args.dataset

        self.start_cache = - args.backward_jump
        self.end_cache = args.forward_jump + args.evaluate_span + 1

        # Grab all data in this environment if scenario is 'specific' or 'no_env':
        if args.cell_type == 'specific' or args.cell_type == 'common':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_env_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                env_id=self.env_id,
                args=args,
                cached=True)
        elif args.cell_type == 'standalone':
            self.image_paths, self.pseudo_paths, self.semantic_mask_paths, self.timestamps, _, _ = get_all_vehicle_frames(
                root_folder=self.root_folder,
                start_index=start_time,
                end_index=end_time,
                vehicle_folder=self.env_id,
                args=args,
                cached=True)

        self.first_timestamp = min(self.timestamps)
        self.last_timestamp = max(self.timestamps)
        self.current_position = self.first_timestamp - 1
        self.BufferSize = args.buffer_size

        self.image_cache = {}
        self.pseudo_cache = {}
        self.semantic_mask_cache = {}
        self.timestamp_cache = {}

        self.img_transform = transforms.Compose([
            #transforms.Resize(COMMON_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])
        self.gt_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).long())
            # transforms.Lambda(lambda x: torch.tensor(x.detach(), dtype=torch.float32)),
        ])
        self.pgt_transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x).long())
            # transforms.Lambda(lambda x: torch.tensor(x.detach(), dtype=torch.float32)),
        ])

        # Load and cache all items with tqdm progress bar
        for idx, img_path in tqdm(enumerate(self.image_paths), desc="Caching Images", total=len(self.image_paths)):
            with Image.open(img_path[0]) as file:
                file = self.img_transform(file)[:3]
                self.image_cache[idx] = copy.copy(file)

        for idx, pgt_path in tqdm(enumerate(self.pseudo_paths), desc="Caching Pseudo Labels",
                                  total=len(self.pseudo_paths)):
            with np.load(pgt_path[0]) as file:
                file = file['arr_0']
                file = self.pgt_transform(file)
                self.pseudo_cache[idx] = copy.copy(file)

        for idx, gt_path in tqdm(enumerate(self.semantic_mask_paths), desc="Caching Semantic Masks",
                                 total=len(self.semantic_mask_paths)):
            with np.load(gt_path[0]) as file:
                file = file['arr_0']
                file = self.gt_transform(file)
                self.semantic_mask_cache[idx] = copy.copy(file)

        for idx, ts in tqdm(enumerate(self.timestamps), desc="Caching Timestamps", total=len(self.timestamps)):
            self.timestamp_cache[idx] = ts

    def __len__(self):
        return len(self.image_paths)

    def create_testset(self, start, stop):
        tss = cached_get_subsets_period(start,
                                        stop,
                                        self.timestamps)
        if len(tss) <= 0:
            return None

        # Obtain the indices of the timestamps in tss
        indices = [i for i, timestamp in enumerate(self.timestamps) if timestamp in tss]

        # Use the indices to obtain the corresponding image, semantic_mask, and pseudo subsets
        imgs_p = [self.image_paths[i] for i in indices]
        imgs = [self.image_cache[i] for i in indices]
        pgts = [self.pseudo_cache[i] for i in indices]
        gts = [self.semantic_mask_cache[i] for i in indices]

        dataset = CachedTorchDataset(imgs, imgs_p, pgts, gts, tss, self.args, test=True)

        return (DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers,
                           pin_memory=False))

    def create_custom_set(self, start, length):
        tss = cached_get_subsets_length(start,
                                        length,
                                        self.timestamps)

        if len(tss) <= 0:
            return None

        # Obtain the indices of the timestamps in tss
        indices = [i for i, timestamp in enumerate(self.timestamps) if timestamp in tss]

        # Use the indices to obtain the corresponding image, semantic_mask, and pseudo subsets
        imgs_p = [self.image_paths[i] for i in indices]
        imgs = [self.image_cache[i] for i in indices]
        pgts = [self.pseudo_cache[i] for i in indices]
        gts = [self.semantic_mask_cache[i] for i in indices]

        dataset = CachedTorchDataset(imgs, imgs_p, pgts, gts, tss, self.args, test=True)
        return (DataLoader(dataset, batch_size=self.args.batchsize, shuffle=False, num_workers=self.args.numworkers,
                           pin_memory=False))

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def create_trainset(self):
        pass

class CachedTorchDataset(Dataset):

    def __init__(self, imgs, imgs_p, pgts, gts, tss, args, test=False):
        self.image = imgs
        self.image_paths = imgs_p
        self.pseudo = pgts
        self.semantic_mask = gts
        self.timestamps = tss
        self.args = args

        self.use_gt = False  # self.args.groundtruth if test else self.args.teacher

    def __getitem__(self, index):
        # Load image and apply transformation
        image = self.image[index]

        # Load pseudo labels and apply transformation
        pseudo = self.pseudo[index]

        # Load semantic masks and apply transformation
        mask = self.semantic_mask[index]

        # Load timestamp
        timestamp = self.timestamps[index]  # [0]

        path = self.image_paths[index][0]

        # Return sample as a tuple
        # print(image.shape, pseudo.shape, mask.shape, timestamp, img_path)
        return image, pseudo, mask, timestamp, path

    def __len__(self):
        return len(self.image)
