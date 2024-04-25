from collections import deque
from torch.utils.data import Dataset, DataLoader
from datasets.dataset import *
from datasets.utils import *


class FifoDataset(OnlineDataset):

    def __init__(self, args, env_id, start_time, end_time, **kwargs):
        super().__init__(args, env_id, start_time, end_time)

        self.ImageBuffer = deque(maxlen=self.BufferSize)
        self.PseudoBuffer = deque(maxlen=self.BufferSize)
        self.SemanticMaskBuffer = deque(maxlen=self.BufferSize)
        self.TimestampsBuffer = deque(maxlen=self.BufferSize)
        self.is_different = False

    def update_queues(self, imgs, pgts, gts, tss):
        self.ImageBuffer.extend(imgs), self.PseudoBuffer.extend(pgts), self.SemanticMaskBuffer.extend(
            gts), self.TimestampsBuffer.extend(tss)
        if len(imgs) > 0:
            self.is_different = True

    def get_queue_content(self):
        return list(self.ImageBuffer), list(self.PseudoBuffer), list(self.SemanticMaskBuffer), list(self.TimestampsBuffer)

    def create_trainset(self):

        imgs, pgts, gts, tss = get_subsets_period(max(0, self.current_position - self.train_step),
                                                  self.current_position,
                                                  self.image_paths,
                                                  self.pseudo_paths,
                                                  self.semantic_mask_paths,
                                                  self.timestamps)

        if len(imgs)>0:
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
        imgs, pgts, gts, tss = self.get_queue_content()
        if len(imgs) < 1 or len(pgts) < 1 or len(gts) < 1 or len(tss) < 1:
            return None
        if len(imgs) != len(pgts) != len(gts) != len(tss):
            return None
        dataset = TorchDataset(imgs, pgts, gts, tss, self.args)

        if len(dataset) < 1:
            return None
        if len(dataset) < self.BufferSize:
            print('# images in buffer :', len(dataset))
        return DataLoader(dataset, batch_size=self.args.batchsize, shuffle=True, num_workers=self.args.numworkers,
                           pin_memory=False)

    def step(self):
        self.current_position += self.train_step
        self.is_different = False


class CachedFifoDataset(CachedOnlineDataset):

    def __init__(self, args, env_id, start_time, end_time):
        super().__init__(args, env_id, start_time, end_time)

        self.ImagePathBuffer = deque(maxlen=self.BufferSize)
        self.ImageBuffer = deque(maxlen=self.BufferSize)
        self.PseudoBuffer = deque(maxlen=self.BufferSize)
        self.SemanticMaskBuffer = deque(maxlen=self.BufferSize)
        self.TimestampsBuffer = deque(maxlen=self.BufferSize)

    def update_queues(self, imgs, imgs_p, pgts, gts, tss):
        self.ImageBuffer.extend(imgs), self.ImagePathBuffer.extend(imgs_p), self.PseudoBuffer.extend(pgts), self.SemanticMaskBuffer.extend(
            gts), self.TimestampsBuffer.extend(tss)

    def get_queue_content(self):
        return list(self.ImageBuffer), list(self.PseudoBuffer), list(self.SemanticMaskBuffer), list(self.TimestampsBuffer), list(self.ImagePathBuffer)

    def create_trainset(self):
        tss = cached_get_subsets_period(max(0, self.current_position - self.train_step),
                                        self.current_position,
                                        self.timestamps)
        if len(tss)>0:
            indices = [i for i, timestamp in enumerate(self.timestamps) if timestamp in tss]

            # Use the indices to obtain the corresponding image, semantic_mask, and pseudo subsets
            imgs_p = [self.image_paths[i] for i in indices]
            imgs = [self.image_cache[i] for i in indices]
            pgts = [self.pseudo_cache[i] for i in indices]
            gts = [self.semantic_mask_cache[i] for i in indices]
            zipped_lists = list(zip(tss, imgs, pgts, gts, imgs_p))
            # Sort the zipped list based on tss values (first element in the tuple)
            zipped_lists.sort()
            # Unzip the sorted list back to individual lists
            tss, imgs, pgts, gts, imgs_p = zip(*zipped_lists)
        # Convert back to lists, if necessary (the unzip step returns tuples)
            tss = list(tss)
            imgs = list(imgs)
            pgts = list(pgts)
            gts = list(gts)
            imgs_p = list(imgs_p)

            self.update_queues(imgs=imgs, imgs_p=imgs_p,  pgts=pgts, gts=gts, tss=tss)

        imgs, pgts, gts, tss, imgs_p = self.get_queue_content()

        dataset = CachedTorchDataset(imgs, imgs_p, pgts, gts, tss, self.args)

        if len(dataset) < 1:
            return None

        if len(dataset) < self.BufferSize:
            print('# images in buffer :', len(dataset))
        return DataLoader(dataset, batch_size=self.args.batchsize, shuffle=True, num_workers=self.args.numworkers,
                           pin_memory=False)

    def step(self):
        self.current_position += self.train_step
