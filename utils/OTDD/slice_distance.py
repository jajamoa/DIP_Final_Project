import sys
import os
import warnings

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from otdd import *
from plot import *
from otdd_pytorch import *

import numpy as np
import json
import cv2
import dataset
import time
import configparser

from PIL import Image,ImageFilter,ImageDraw

def loadData(root,classes=3,n=20):
    data_list=os.listdir(root)
    result=[]
    labels=[]
    for i in range(classes):
        current_num=0
        for d in data_list:
            if int(d[-5])==i:
                current_num+=1
                result.append(os.path.join(root,d))
                labels.append(i)
            if current_num>=n:
                break
    features,labels=loadDataSet(result,labels)
    return features,labels
                
    
def loadImg(img_path):

    normalize =   transforms.Normalize((0.1307,), (0.3081,))
    
    image = Image.open(img_path).convert('L')
    
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        normalize
    ])

    image = transform(image)

    image =np.array(image)
    
    return image

def loadDataSet(data_list,labels):
    dataset=[]
    for img in data_list:
        dataset.append(loadImg(img).reshape(-1))
    dataset=torch.tensor(dataset, dtype=torch.float32)
    labels=torch.tensor(labels)
    return dataset,labels
    
def dataEnhance(features):
    

if __name__=='__main__':

    DATA_DIR='/home/mist/data/first_classifier/train'

    
    classes=[0,1,2]
    
    features,labels=loadData(DATA_DIR)
    

    features_he=dataEnhance(features)
    
    
    distance_tensorized = PytorchEuclideanDistance()
    routine_tensorized = SinkhornTensorized(distance_tensorized)
    cost_tensorized = SamplesLossTensorized(routine_tensorized)

    outputs = cost_tensorized.distance(features, features_he, return_transport=True)

    print(outputs[0])

    fig,_=plot_coupling(outputs[1].numpy(), outputs[2].numpy(), labels.numpy(), labels_he.numpy(),
                      classes, classes, figsize=(10,10), cmap='OrRd')
    fig.savefig('FeatureDistance.png')
    
        
    outputs2 = cost_tensorized.distance_with_labels(features, features_he, 
                                              labels, labels_he, gaussian_class_distance=False)
    
    fig,_=plot_coupling(outputs2[1].numpy(), 
              outputs2[2].numpy(), 
              labels.numpy(), 
              labels_he.numpy(),
                  classes, classes, figsize=(10,10), cmap='OrRd')

    fig.savefig('LabelDistance.png')
    

    fig,_=plot_class_distances(outputs2[3], classes, classes, 
                                cmap='OrRd', figsize=(10,8), text=True)

    fig.savefig('ClassDistance.png')
    
#     distance_function = POTDistance(distance_metric='euclidean')

#     cost_function = SinkhornCost(distance_function, 0.02)

#     cost, coupling, M_dist = cost_function.distance(features, features_he)

#     fig,_=plot_coupling(coupling, M_dist,labels, labels_he,
#                       classes, classes, figsize=(5,5), cmap='OrRd')
    
#     fig.savefig('Coupling.png')

#     plot=plot_coupling_network(features, features_he, 
#                                labels, 
#                                 labels_he, coupling, plot_type='ds')
    
#     plot.savefig('Coupling_Network.png')