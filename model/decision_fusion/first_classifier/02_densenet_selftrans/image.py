import random
import os
import math
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
import torch
from torchvision import transforms

def load_data(img_path, train = True):


    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                        std=[0.33165374, 0.33165374, 0.33165374])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(90),
        # random brightness and random contrast
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])


    image = Image.open(img_path).convert('RGB')
    image = transform(image)

    image =np.array(image)
    
    label_index = int(img_path.split('.')[0][-1])
    assert label_index in [0, 1, 2]
    ones = torch.sparse.torch.eye(3)
    label = ones.index_select(0, torch.tensor([label_index]))
    return image, label
