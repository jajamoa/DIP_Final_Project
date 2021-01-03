import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
from torchvision import transforms
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, train=False):
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.train = train

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'         
        img_path = self.lines[index]
        img, label = load_data(img_path,self.train)
        return img, label


class mocoDataset(Dataset):
    def __init__(self, root, train=False , aug_plus=True):
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.aug_plus = aug_plus

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'         
        img_path = self.lines[index]
        imgs, label = load_data_moco(img_path,aug_plus=self.aug_plus)
        return imgs, label
