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
    if train:
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
    else:
        transform = transforms.Compose([
        #     transforms.Resize(224),
        #     transforms.CenterCrop(224),
            transforms.Resize((224,224)),
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


def load_data_moco(img_path, train = True , aug_plus=True):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    transform = TwoCropsTransform(transforms.Compose(augmentation))

    image = Image.open(img_path).convert('RGB')
    img_q,img_k = transform(image)
    images = torch.tensor([np.array(img_q),np.array(img_k)])
    label_index = int(img_path.split('.')[0][-1])
    assert label_index in [0, 1, 2]
    ones = torch.sparse.torch.eye(3)
    label = ones.index_select(0, torch.tensor([label_index]))
    return images, label



class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
