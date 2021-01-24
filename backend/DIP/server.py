import sys
import os
import warnings
from model import COVNet_CSR
from utils import load_net
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

import numpy as np
import argparse
import json
import cv2
import dataset
import time
import configparser

from flask import Flask
app = Flask(__name__)


target_names=["Non-infected","Cap","Covid-19"]


parser = argparse.ArgumentParser(description='PyTorch DIP2020')


parser.add_argument('--model', '-m', metavar='MODEL', default=None, type=str,
                    help='path to the model')

parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE' ,type=int,
help='batch size', default=16)

                    
parser.add_argument('--gpu',metavar='GPU', type=str,
                    help='GPU id to use.', default="0")

parser.add_argument('--count', '-c', metavar='IMG_COUNT', type=int,
                    help='Count of imgs.', default=25)


def test(test_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(test_list,train=False,count=args.count), 
    batch_size=args.batch_size, shuffle = False)
    model.eval()
    correct = 0
    
    predlist=[]
    scorelist=[]
    targetlist=[]
    with torch.no_grad():
        for i,(img, target) in enumerate(test_loader):
            target = target[0]
            img = img.cuda()
            img = img.type(torch.FloatTensor)
            output = model(target, img)
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            # print(target, int(pred))
            pred=pred.squeeze(1)

            predlist=np.append(predlist, pred.cpu().numpy())
            scorelist=np.append(scorelist, score.cpu().numpy()[:,1])
            targetlist=np.append(targetlist,target)
            # print("{} ---- {}".format(i,len(test_loader)))
    print ('test done')
    return targetlist, scorelist, predlist

class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, targets, *inputs):
        outputs = self.model(*inputs)
#         loss = self.loss(outputs, torch.topk(targets.long(), 1)[1].squeeze(1))
        return outputs

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


conf= configparser.ConfigParser()
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if not args.model:
    print("Usage: --model -m\n\tpath to the model")
    sys.exit()
    
model = COVNet_CSR(3)
criterion = nn.CrossEntropyLoss().cuda()


model = DataParallel_withLoss(model, criterion)
    
trained_net = torch.load(args.model)
model.load_state_dict(trained_net['state_dict'])
model = model.cuda()


@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/result/<name>/', methods=['GET', 'POST'])
def prefict():
    path = name
    test_list = [TEST_DIR]
    targetlist, scorelist, predlist = test(test_list, model, criterion)
    print(targetlist[0],'--',target_names[int(predlist[0])])
    return jsonify({name:target_names[int(predlist[0])]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)