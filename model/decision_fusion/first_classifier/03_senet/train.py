import sys
import os
import warnings
from senet.se_resnet import se_resnet50
from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import wandb
import argparse
import json
import cv2
import dataset
import time
import configparser
import random


wandb.init(project="DIP Final Project 2020")

parser = argparse.ArgumentParser(description='PyTorch DIP2020')

parser.add_argument('--config', '-c', metavar='CONFIG',type=str,
                    help='path to confg file')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')


parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE' ,type=int,
                    help='batch size', default=2)

parser.add_argument('--k_flod', '-k', metavar='KFOLD' ,type=int,
                    help='k-fold size', default=10)

parser.add_argument('--gpu',metavar='GPU', type=str,
                    help='GPU id to use.', default="0")

parser.add_argument('--task',metavar='TASKID', type=str, default=wandb.run.name, 
                    help='Task id of this run.')

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
    global args, best_prec1
    best_prec1 = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-6
    args.momentum  = 0.95
    args.decay  = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 50
    args.steps = [-1,1,20,40]
    args.scales = [1,1,0.5,0.5]
    args.workers = 0
    args.seed = time.time()
    args.print_freq = 30
    wandb.config.update(args)
    wandb.run.name = f"Default_{wandb.run.name}" if (args.task == wandb.run.name) else f"{args.task}_{wandb.run.name}"

    conf= configparser.ConfigParser()

    conf.read(args.config) 
    TRAIN_DIR = conf.get("senet", "train") 
    VALID_DIR = conf.get("senet", "valid") 
    TEST_DIR = conf.get("senet", "test") 
    LOG_DIR = conf.get("senet", "log") 
    create_dir_not_exist(LOG_DIR)
    train_list = [os.path.join(TRAIN_DIR, item) for item in os.listdir(TRAIN_DIR)]
    train_list=checkSuffix(train_list)
    val_list = [os.path.join(VALID_DIR, item) for item in os.listdir(VALID_DIR)]
    val_list=checkSuffix(val_list)
    data_list= train_list+val_list
    random.shuffle(data_list)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    
    model = se_resnet50(num_classes=3)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    model = DataParallel_withLoss(model, criterion)

    for i in range(args.k_fold):
        train_list,val_list=get_k_fold_data(i,data_list)
        args.lr=args.original_lr
        best_prec1 = 1e6
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay)

        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)
            train_loss = train(train_list, model, criterion, optimizer, epoch)
            prec1 = validate(val_list, model, criterion, epoch)
            with open(os.path.join(LOG_DIR, args.task + ".txt"), "a") as f:
                f.write("K "+str(i) +" epoch " + str(epoch) +  "  TrainLoss: " +str(float(train_loss))+
                "  ValLoss: " +str(float(prec1)))
                f.write("\n")
            wandb.log({'K':i,'epoch': epoch, 'TrainCEloss': train_loss,'ValCEloss':prec1})
            wandb.save(os.path.join(LOG_DIR, args.task + ".txt"))
            is_best = prec1 < best_prec1
            best_prec1 = min(prec1, best_prec1)
            print(' * best CELoss {CELoss:.3f} '.format(CELoss=best_prec1))
            save_checkpoint({
                'k':i,
                'epoch': epoch + 1,
                'arch': args.pre,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best,args.task, epoch = epoch,path=os.path.join(LOG_DIR, args.task))


########k_fold############        
def get_k_fold_data(i, datalist): 
    k=args.k_flod
    assert k > 1
    fold_size = len(datalist) // k  # 每份的个数:数据总条数/折数（组数）
    
    train_list=[]
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step)切片函数
        data_part = datalist[idx]
        if j == i: ###第i折作valid
            valid_list = data_part
        else:
            train_list += data_part
    return train_list,valid_list


def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list,train=True), num_workers=args.workers, batch_size=args.batch_size, shuffle = True)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    

    
    model.train()
    end = time.time()
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        img = img.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor).squeeze(1).cuda()
        # print(img.shape)
        # print(target.shape)
        loss,_ = model(target, img)
        loss = loss.sum()
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {lossval:.4f} ({lossavg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lossval=losses.val/args.batch_size, lossavg=losses.avg/args.batch_size))
    CEloss=losses.avg/args.batch_size
    return CEloss
    

def validate(val_list, model, criterion, epoch):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list,train=False), batch_size=args.batch_size, shuffle = False)
    model.eval()
    CELoss = 0
    with torch.no_grad():
        for i,(img, target) in enumerate(test_loader):
            img = img.cuda()
            img = img.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor).squeeze(1).cuda()
            _, output = model(target, img)
            CELoss += criterion(output.data,torch.topk(target.long(), 1)[1].squeeze(1))
    CELoss = CELoss/len(test_loader)/args.batch_size
    print(' * CELoss {CELoss:.3f} '.format(CELoss=CELoss))
    return CELoss


def adjust_learning_rate(optimizer, epoch):
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    wandb.log({'Learning Rate': args.lr})

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
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
