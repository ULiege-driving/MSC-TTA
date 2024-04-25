import sys
import os
import torch
import cv2
import numpy as np
from collections import deque

from torch.utils.data import Dataset, DataLoader

from datasets.dataset import *
from datasets.utils import *
from training.classic import *
import copy
import random


class MirDataset(OnlineDataset):

    def __init__(self, args, env_id, start_time, end_time, **kwargs):
        super().__init__(args, env_id, start_time, end_time)

        self.model = kwargs['model']
        self.criterion = kwargs['criterion']
        self.optimizer = kwargs['optimizer']
        self.scaler = kwargs['scaler']
        self.Trainer = kwargs['Trainer']

        self.k = args.k
        self.C = args.C

        self.ImageBuffer = []
        self.PseudoBuffer = []
        self.SemanticMaskBuffer = []
        self.TimestampsBuffer = []

        self.is_different = False

    def update_queues(self, imgs, pgts, gts, tss):

        if len(imgs) > 0:
            self.is_different = True
            for img, pgt, gt, ts in zip(imgs, pgts, gts, tss):
                if len(self.ImageBuffer) < self.BufferSize:
                    self.ImageBuffer.append(img), self.PseudoBuffer.append(pgt), self.SemanticMaskBuffer.append(
                        gt), self.TimestampsBuffer.append(ts)

                else:
                    index = random.randrange(self.BufferSize)
                    del self.ImageBuffer[index], self.PseudoBuffer[index], self.SemanticMaskBuffer[index], self.TimestampsBuffer[index]
                    self.ImageBuffer.append(img), self.PseudoBuffer.append(pgt), self.SemanticMaskBuffer.append(
                        gt), self.TimestampsBuffer.append(ts)

    def simulate_training_epoch(self, dataloader: DataLoader):
        model_copy = copy.deepcopy(self.model)
        optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=self.args.lr)
        scaler_copy = torch.cuda.amp.GradScaler()
        trainer = self.Trainer(model_copy, optimizer_copy, scaler_copy, self.criterion, self.args.device, self.args)
        trainer.train_epoch(dataloader)

        return trainer.network

    def retrieve_top_k_samples(self, k, future_model, imgs, pgts, gts, tss):
        interference = []
        dataset = TorchDataset(imgs, pgts, gts, tss, self.args)

        for i in range(len(dataset)):
            torch.cuda.empty_cache()

            # compute interference associated to that sample
            img, pgt, gt, _, _ = dataset[i]
            if self.args.supervision == 'teacher':
                target = pgt
            else:
                target = gt

            with torch.cuda.amp.autocast():
                img = img.unsqueeze(0).to(self.args.device)
                target = target.unsqueeze(0).to(self.args.device)
                pre_loss = self.criterion(self.model.forward(img), target).item()
                post_loss = self.criterion(future_model.forward(img), target).item()

            interference.append([post_loss - pre_loss, i])

        # Sort in the descending order
        interference = np.array(interference)
        interference = interference[interference[:, 0].argsort()][::-1]

        return interference[:, 1][:k]

    def retrieve_c_samples(self, C, imgs, pgts, gts, tss):

        N = len(imgs)

        return np.array(random.sample(range(N), C))

    def get_queue_content(self):
        return self.ImageBuffer.copy(), self.PseudoBuffer.copy(), self.SemanticMaskBuffer.copy(), self.TimestampsBuffer.copy()

    def create_trainset(self):
        imgs, pgts, gts, tss = get_subsets_period(max(0, self.current_position - self.train_step),
                                                  self.current_position,
                                                  self.image_paths,
                                                  self.pseudo_paths,
                                                  self.semantic_mask_paths,
                                                  self.timestamps)
        if len(imgs) > 0:
            zipped_lists = list(zip(tss, imgs, pgts, gts))
            # Sort the zipped list based on tss values (first element in the tuple)
            zipped_lists.sort()
            # Unzip the sorted list back to individual lists
            tss, imgs, pgts, gts = zip(*zipped_lists)
        # Convert back to lists, if necessary (the unzip step returns tuples)
        tss = list(tss)
        imgs = list(imgs)
        pgts = list(pgts)
        gts = list(gts)

        self.update_queues(imgs=imgs, pgts=pgts, gts=gts, tss=tss)

        if len(imgs) < 1 or len(pgts) < 1 or len(gts) < 1 or len(tss) < 1:
            return None
        if len(imgs) != len(pgts) != len(gts) != len(tss):
            return None
        temp_dataset = TorchDataset(imgs, pgts, gts, tss, self.args)
        temp_dataloader = DataLoader(temp_dataset, batch_size=self.args.batchsize, shuffle=True, num_workers=self.args.numworkers,
                           pin_memory=False)
        future_model = self.simulate_training_epoch(temp_dataloader)
        print("TEMP: {}".format(len(temp_dataset)))
        k = None


        t_imgs, t_pgts, t_gts, t_tss = self.get_queue_content()
        if len(t_imgs) > self.C:
            ids = self.retrieve_c_samples(self.C, t_imgs, t_pgts, t_gts, t_tss)
            ids = sorted(ids.astype('int').tolist())
            print(len(ids))
            t_imgs = [t_imgs[i] for i in ids]
            t_pgts = [t_pgts[i] for i in ids]
            t_gts = [t_gts[i] for i in ids]
            t_tss = [t_tss[i] for i in ids]

        if len(t_imgs) <= self.k:
            k = len(t_imgs)
        else:
            k = self.k

        print(k)
        if k > 0:
            ids = self.retrieve_top_k_samples(k, future_model, t_imgs, t_pgts, t_gts, t_tss)
            ids = sorted(ids.astype('int').tolist())
            print(len(ids))
            t_imgs = [t_imgs[i] for i in ids]
            t_pgts = [t_pgts[i] for i in ids]
            t_gts = [t_gts[i] for i in ids]
            t_tss = [t_tss[i] for i in ids]


        if len(t_imgs) < 1 or len(t_pgts) < 1 or len(t_gts) < 1 or len(t_tss) < 1:
            return None
        if len(t_imgs) != len(t_pgts) != len(t_gts) != len(t_tss):
            return None
        dataset = TorchDataset(t_imgs, t_pgts, t_gts, t_tss, self.args)

        if len(dataset) < 1:
            return None
        if len(dataset) < self.BufferSize:
            print('# images in buffer :', len(self.ImageBuffer))
        return DataLoader(dataset, batch_size=self.args.batchsize, shuffle=True, num_workers=self.args.numworkers,
                           pin_memory=False)

    def step(self):
        self.current_position += self.train_step
        self.is_different = False


