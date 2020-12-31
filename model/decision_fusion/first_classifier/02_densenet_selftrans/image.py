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
    transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform2 = transforms.ToTensor()

    image = np.array(Image.open(img_path).convert('RGB'))
    image = transform(image)
    label_index = int(img_path.split('.')[0][-1])
    assert label_index in [0, 1, 2]
    ones = torch.sparse.torch.eye(3)
    label = ones.index_select(0, torch.tensor([label_index]))
    return image, label
