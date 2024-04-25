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


class UniformDataset(OnlineDataset):

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


    def retrieve_k_samples(self, k, imgs, pgts, gts, tss):

        N = len(imgs)

        return np.array(random.sample(range(N), k))

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

        k = None


        if len(self.ImageBuffer) < self.k:
            k = len(self.ImageBuffer)
            t_imgs, t_pgts, t_gts, t_tss = self.get_queue_content()
        else:
            k = self.k
            t_imgs, t_pgts, t_gts, t_tss = self.get_queue_content()
            """
            random_indexes = sorted(random.sample(range(len(t_imgs)), self.C))
            t_imgs = [t_imgs[i] for i in random_indexes]
            t_pgts = [t_pgts[i] for i in random_indexes]
            t_gts = [t_gts[i] for i in random_indexes]
            t_tss = [t_tss[i] for i in random_indexes]
            """

        if k > 0:
            ids = self.retrieve_k_samples(k, t_imgs, t_pgts, t_gts, t_tss)
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


