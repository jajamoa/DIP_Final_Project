import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy import ndimage
from skimage import measure
from scipy import stats

import sys
import os
import warnings

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import SimpleITK as sitk
import json
import cv2
import time
import configparser

from torchvision import transforms

from PIL import Image,ImageFilter,ImageDraw

def create_dir_not_exist(path):
    for length in range(1, len(path.split(os.path.sep))):
        check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
        if not os.path.exists(check_path):
            os.mkdir(check_path)
            print(f'Created Dir: {check_path}')

def segmentData(root,root_result):
    data_list=os.listdir(root)
    for d in data_list:
        patient=os.path.join(root,d)
        slice_list=os.listdir(patient)
        create_dir_not_exist(os.path.join(root_result,d))
        for slice in slice_list:
            img=loadImg(os.path.join(patient,slice))
            segmentation = mask.apply(img)
#             segmentation = segmentation.reshape(segmentation.shape[1],segmentation.shape[2])
#             cv2.imwrite(os.path.join(root_result,d,slice),segmentation)
            fl =segmentation.flatten()
            for p in fl:
                if p>0:
                    print(p)
            sys.exit()

        
                
    
def loadImg(img_path):

    
    image = Image.open(img_path).convert('RGB')
    
    image =np.array(image)
#     image = image.reshape(1,image.shape[0],image.shape[1]) 

    
    
    return image

def loadDataSet(data_list,labels):
    dataset=[]
    for img in data_list:
        dataset.append(loadImg(img).reshape(-1))
    dataset=torch.tensor(dataset, dtype=torch.float32)
    labels=torch.tensor(labels)
    return dataset,labels

    #####################################################
# Generate binary structure to mimic trachea
#####################################################

def generate_structure_trachea(Radius, RadiusX, RadiusZ):
    
    struct_trachea = np.zeros([2*Radius+1,2*Radius+1,RadiusZ])
    for i in range(0,2*Radius+1):
        for j in range(0,2*Radius+1):
            if distance.euclidean([Radius+1,Radius+1],[i,j]) < RadiusX:
                struct_trachea[i,j,:] = 1
            else:
                struct_trachea[i,j,:] = 0
    
    return struct_trachea

#####################################################
# Generate bounding box
#####################################################

def bbox2_3D(img,label,margin,limit):
    
    imgtmp = np.zeros(img.shape)
    imgtmp[img == label] = 1
    
    x = np.any(imgtmp, axis=(1, 2))
    y = np.any(imgtmp, axis=(0, 2))
    z = np.any(imgtmp, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    xmin = xmin - margin - 1
    xmin = max(0,xmin)
    ymin = ymin - margin - 1
    ymin = max(0,ymin)
    zmin = zmin - margin - 1
    zmin = max(0,zmin)        
    xmax = xmax + margin + 1
    xmax = min(xmax,limit[0])
    ymax = ymax + margin + 1
    ymax = min(ymax,limit[1])
    zmax = zmax + margin + 1
    zmax = min(zmax,limit[2])
        
    return xmin, xmax, ymin, ymax, zmin, zmax
    
    segmentData(DATA_DIR,RESULT_DIR)



if __name__=='__main__':
    
    DATA_DIR='/home/mist/data/testData'
    RESULT_DIR = '/home/mist/data/segment'
    
    
    params = {}

    #####################################################
    # Parameters for intensity (fixed)
    #####################################################

    params['lungMinValue']      = -1024
    params['lungMaxValue']      = -400
    params['lungThreshold']     = -900

    #####################################################
    # Parameters for lung segmentation (fixed)
    #####################################################

    params['xRangeRatio1']      = 0.4
    params['xRangeRatio2']      = 0.75
    params['zRangeRatio1']      = 0.5
    params['zRangeRatio2']      = 0.75

    #####################################################
    # Parameters for airway segmentation
    # NEED TO ADAPT for image resolution and orientation 
    #####################################################
    params['airwayRadiusMask']  = 15  # increase the value if you have high resolution image
    params['airwayRadiusX']     = 8   # ditto
    params['airwayRadiusZ']     = 15  # ditto
    params['super2infer']       = 0   # value = 1 if slice no. increases from superior to inferior, else value = 0
    
    

        #####################################################
    # Load image 
    #####################################################
    I         = loadImg("/home/mist/data/testData/T000/CT0000.png")

    #####################################################
    # Intensity thresholding & Morphological operations
    #####################################################

    M = np.zeros(I.shape)
    M[I > params['lungMinValue']] = 1
    M[I > params['lungMaxValue']] = 0

    struct_s = ndimage.generate_binary_structure(3, 1)
    struct_m = ndimage.iterate_structure(struct_s, 2)
    struct_l = ndimage.iterate_structure(struct_s, 3)
    M = ndimage.binary_closing(M, structure=struct_s, iterations = 1)
    M = ndimage.binary_opening(M, structure=struct_m, iterations = 1)

    #####################################################
    # Estimate lung filed of view
    #####################################################

    [m, n, p] = I.shape;
    medx      = int(m/2)
    medy      = int(n/2)
    xrange1   = int(m/2*params['xRangeRatio1'])
    xrange2   = int(m/2*params['xRangeRatio2'])
    zrange1   = int(p*params['zRangeRatio1'])
    zrange2   = int(p*params['zRangeRatio2'])

    #####################################################
    # Select largest connected components & save nii
    #####################################################

    M = measure.label(M)
    label1 = M[medx - xrange2 : medx - xrange1, medy, zrange1 : zrange2]
    label2 = M[medx + xrange1 : medx + xrange2, medy, zrange1 : zrange2]
    label1 = stats.mode(label1[label1 > 0])[0][0]
    label2 = stats.mode(label2[label2 > 0])[0][0]
    M[M == label1] = -1
    M[M == label2] = -1
    M[M > 0] = 0
    M = M*-1

#     M     = ndimage.binary_closing(M, structure = struct_m, iterations = 1)
#     M     = ndimage.binary_fill_holes(M)
#     Mlung = np.int8(M)
#     nib.Nifti1Image(Mlung,I_affine).to_filename('./result/sample_lungaw.nii.gz')

    #####################################################
    # Display segmentation results 
    #####################################################

    plt.figure(1)
    slice_no = int(p/2)
    plt.subplot(121)
    plt.imshow(np.fliplr(np.rot90(I[:,:,slice_no])), cmap = plt.cm.gray)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(np.fliplr(np.rot90(Mlung[:,:,slice_no])), cmap = plt.cm.gray)
    plt.axis('off')

    plt.figure(2)
    slice_no = int(n*0.5)
    plt.subplot(121)
    plt.imshow(np.fliplr(np.rot90(I[:,slice_no,:])), cmap = plt.cm.gray)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(np.fliplr(np.rot90(Mlung[:,slice_no,:])), cmap = plt.cm.gray)
    plt.axis('off')