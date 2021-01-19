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

def load_data(folder_path, train = True):
    transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform2 = transforms.ToTensor()

    first = True

    img_path_list = []
    for img_path in os.listdir(folder_path):
        if (img_path[0] == '.'):
            continue
        img_path_list.append(img_path)

    for img_path in img_path_list[:20]:
        image = np.array(Image.open(os.path.join(folder_path, img_path)).convert('RGB'))
        image = transform(image)
        image = image.unsqueeze(dim = 0)
        # print(image.size())
        if first:
            image_set = image
            first = False
        else:
            image_set =  torch.cat([image_set, image], axis = 0)
            # print(image_set.size())

    label_index = int(folder_path.split(os.path.sep)[-1][0])
    assert label_index in [0, 1, 2]
    ones = torch.sparse.torch.eye(3)
    label = ones.index_select(0, torch.tensor([label_index]))
    return image_set, label
