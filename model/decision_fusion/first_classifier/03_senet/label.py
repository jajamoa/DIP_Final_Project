import sys
import os
import warnings
from senet.se_resnet import se_resnet50
from utils import load_net

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

import numpy as np
import wandb
import argparse
import json
import cv2
import dataset
import time
import configparser

import csv


parser = argparse.ArgumentParser(description='PyTorch DIP2020')

parser.add_argument('--config', '-c', metavar='CONFIG',type=str,
                    help='path to confg file')


parser.add_argument('--model', '-m', metavar='MODEL', default=None, type=str,
                    help='path to the model')

parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE' ,type=int,
help='batch size', default=32)

                    
parser.add_argument('--gpu',metavar='GPU', type=str,
                    help='GPU id to use.', default="0")

CLASSES_NAME = ["Non-infected","Cap","Covid-19"]


def create_dir_not_exist(path):
    for length in range(1, len(path.split(os.path.sep))):
        check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
        if not os.path.exists(check_path):
            os.mkdir(check_path)
            print(f'Created Dir: {check_path}')

def checkSuffix(file_list):
    img_suffixs=['png']
    for file in file_list:
        if not (file.split('.')[-1] in img_suffixs):
            file_list.remove(file)
    return file_list

def main():
    global args
    conf= configparser.ConfigParser()
    args = parser.parse_args()

    conf.read(args.config) 
    DATA_DIR = conf.get("subject_level", "data") 
    LABEL_DIR = conf.get("subject_level", "label") 
    create_dir_not_exist(LABEL_DIR)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.model:
        print("Usage: --model -m\n\tpath to the model")
        sys.exit()
        
    model = se_resnet50(num_classes=3)  


    
    model = DataParallel(model)
        
    trained_net = torch.load(args.model)
    model.load_state_dict(trained_net['state_dict'])
    model = model.cuda()
    
    result_list=[]
    

    for class_index,class_name in enumerate(CLASSES_NAME):
        
        class_dir = os.path.join(DATA_DIR,class_name)
        patient_list = os.listdir(class_dir)
        patient_list = [os.path.join(class_dir,patient) for patient in patient_list 
                        if os.path.isdir(os.path.join(class_dir,patient)) ]
        print('---------- {} ----------'.format(class_name))
        for i,patient_dir in enumerate(patient_list):
            slice_list = os.listdir(patient_dir)
            if len(slice_list)<10:
                continue
            checkSuffix(slice_list)
            slice_list = [s for s in slice_list if s[:2] != "._" ]
            slice_list = [ os.path.join(patient_dir,s) for s in slice_list ]
            scorelist= test(slice_list,class_index,model)
            result = np.insert(scorelist,0,class_index)
            print('{} ----- {}'.format(i,len(patient_list)))
            result_list.append(list(result))
            
            
    with open(os.path.join(LABEL_DIR,'result.csv'),'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerows(result_list)
    




def test(test_list, class_index , model):

    test_loader = torch.utils.data.DataLoader(dataset.subjectDataset(test_list,class_index), 
    batch_size=args.batch_size, shuffle = False)
    model.eval()

    scorelist=[]

    with torch.no_grad():
        for i,(img, _) in enumerate(test_loader):
            img = img.cuda()
            img = img.type(torch.FloatTensor)
            output = model( img)
            score = F.softmax(output, dim=1)

            scorelist=np.append(scorelist, score.cpu().numpy())

    return scorelist

class FullModel(nn.Module):
    def __init__(self, model):
        super(FullModel, self).__init__()
        self.model = model


    def forward(self, *inputs):
        outputs = self.model(*inputs)
        return outputs

def DataParallel(model,**kwargs):
    model=FullModel(model)
    if 'device_ids' in kwargs.keys():
        device_ids=kwargs['device_ids']
    else:
        device_ids=None
    if 'output_device' in kwargs.keys():
        output_device=kwargs['output_device']
    else:
        output_device=None
    if 'cuda' in kwargs.keys():
        cudaID=kwargs['cuda']
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda()
    return model


if __name__ == '__main__':
    main()
