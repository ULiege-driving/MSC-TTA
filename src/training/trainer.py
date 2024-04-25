import sys
import os
sys.path.append(os.path.abspath('..'))
import torch
import numpy as np


from abc import abstractmethod
from torch.utils.data import DataLoader

from evaluate.evaluation import SegmentationEvaluator


class NetworkTrainer():

    def __init__(self, network, optimizer, scaler, criterion, device, args):
        self.args = args
        self.network = network
        self.optimizer = optimizer
        self.scaler = scaler
        self.criterion = criterion
        self.device = device

        self.image_counter = 0

    @abstractmethod
    def train_epoch(self, dataloader: DataLoader):
        pass