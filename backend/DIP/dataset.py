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
    def __init__(self, root, count=25,  train=False):
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.train = train
        self.count = count

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'         
        img_path = self.lines[index]
        img, label = load_data(img_path, n = self.count, train = self.train)
        return img, label
