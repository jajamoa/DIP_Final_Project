import sys
import os
import warnings
from model import resnet
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


parser = argparse.ArgumentParser(description='PyTorch DIP2020')

parser.add_argument('--config', '-c', metavar='CONFIG',type=str,
                    help='path to confg file')


parser.add_argument('--model', '-m', metavar='MODEL', default=None, type=str,
                    help='path to the model')

parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE' ,type=int,
help='batch size', default=16)

                    
parser.add_argument('--gpu',metavar='GPU', type=str,
                    help='GPU id to use.', default="0")


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
    TEST_DIR = conf.get("resnet", "test") 
    LOG_DIR = conf.get("resnet", "log") 
    create_dir_not_exist(LOG_DIR)
    test_list = [os.path.join(TEST_DIR, item) for item in os.listdir(TEST_DIR)]
    test_list=checkSuffix(test_list)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not args.model:
        print("Usage: --model -m\n\tpath to the model")
        sys.exit()
        
    model = resnet()
    criterion = nn.CrossEntropyLoss().cuda()

    
    model = DataParallel_withLoss(model, criterion)
        
    trained_net = torch.load(args.model)
    model.load_state_dict(trained_net['state_dict'])
    model = model.cuda()


    vote_pred = np.zeros(len(test_list))
    vote_score = np.zeros(len(test_list))

    targetlist, scorelist, predlist = test(test_list,model,criterion)


    report = classification_report(y_true=targetlist, y_pred=predlist,target_names=["Normal","CAP","COVID-19"])
    print(report)


def test(test_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(test_list,train=False), 
    batch_size=args.batch_size, shuffle = False)
    model.eval()
    correct = 0
    
    predlist=[]
    scorelist=[]
    targetlist=[]
    with torch.no_grad():
        for i,(img, target) in enumerate(test_loader):
            img = img.cuda()
            img = img.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor).squeeze(1).cuda()
            _, output = model(target, img)
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            pred=pred.squeeze(1)
            target=torch.topk(target.long(), 1)[1].squeeze(1)

            correct += pred.eq(target).sum().item()
            

            targetcpu=target.cpu().numpy()
            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,targetcpu)
            print("{} ---- {}".format(i,len(test_loader)))
    print ('test done')
    return targetlist, scorelist, predlist

class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, targets, *inputs):
        outputs = self.model(*inputs)
        loss = self.loss(outputs, torch.topk(targets.long(), 1)[1].squeeze(1))
        return torch.unsqueeze(loss, 0), outputs

def DataParallel_withLoss(model,loss,**kwargs):
    model=FullModel(model, loss)
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
