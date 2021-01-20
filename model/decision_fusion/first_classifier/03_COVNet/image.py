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
from batchgenerators.transforms import noise_transforms
from batchgenerators.transforms import spatial_transforms

def do_augmentation(array):
        """Augmentation for the training data.
        :array: A numpy array of size [c, x, y, z]
        :returns: augmented image and the corresponding mask
        """
        # normalize image to range [0, 1], then apply this transform
        patch_size = np.asarray(array.shape)[1:]
        augmented = noise_transforms.augment_gaussian_noise(
            array, noise_variance=(0, .015))

        # need to become [bs, c, x, y, z] before augment_spatial
        augmented = augmented[None, ...]
        # mask = mask[None, None, ...]
        r_range = (0, (3 / 360.) * 2 * np.pi)
        cval = 0.

        augmented, _ = spatial_transforms.augment_spatial(
            augmented, patch_size=patch_size, seg=None,
            # do_elastic_deform=True, alpha=(0., 100.), sigma=(8., 13.),
            do_rotation=True, angle_x=r_range, angle_y=r_range, angle_z=r_range,
            do_scale=True, scale=(.9, 1.1),
            border_mode_data='constant', border_cval_data=cval,
            order_data=3,
            # p_el_per_sample=0.5,
            p_scale_per_sample=.5,
            p_rot_per_sample=.5,
            random_crop=False
        )
        return augmented[0]

def graygate(I,threshold):
    I=np.array(I.convert('L'))
    I = I.reshape(-1)
    p_max=np.max(I)
    i=0
    for p in I:
        if p==p_max:
            i+=1
    return int(i>threshold)

def load_data(folder_path, n ,train = False):
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform2 = transforms.ToTensor()

    first = True

    img_path_list = []
    for img_path in os.listdir(folder_path):
        if (img_path[0] == '.'):
            continue
        img_path_list.append(img_path)
    
    if len(img_path_list) > n:
        total = len(img_path_list)
        step = int(np.floor(total / n))
        img_path_list = img_path_list[:step*n:step]

    for img_path in img_path_list:
        image = Image.open(os.path.join(folder_path, img_path)).convert('RGB')
        if first:
            gray_class=graygate(image,1000)
        image = transform(image)
        image = image.unsqueeze(dim = 0)
        # print(image.size())
        if first:
            image_set = image
            first = False
        else:
            image_set =  torch.cat([image_set, image], axis = 0)
            # print(image_set.size())

    if train:
        image_set = torch.FloatTensor(do_augmentation(np.array(image_set)))
    label_index = int(folder_path.split(os.path.sep)[-1][0])
    assert label_index in [0, 1, 2]
    ones = torch.sparse.torch.eye(3)
    label = ones.index_select(0, torch.tensor([label_index]))
    return image_set, label , gray_class
