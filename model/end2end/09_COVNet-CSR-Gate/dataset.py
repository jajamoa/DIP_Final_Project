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
    def __init__(self, root, train=False, n=20):
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.train = train
        self.n = n

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'         
        img_path = self.lines[index]
        img, label,gray_classes = load_data(img_path, train = self.train ,n=self.n )
        return img, label ,gray_classes