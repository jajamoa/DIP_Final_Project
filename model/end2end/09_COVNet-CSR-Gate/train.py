import sys
import os
import warnings
from model import COVNet_CSR_Gate
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

wandb.init(project="DIP Final Project 2020")

parser = argparse.ArgumentParser(description='PyTorch DIP2020')

parser.add_argument('--config', '-c', metavar='CONFIG',type=str,
                    help='path to confg file')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')

parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE' ,type=int,
                    help='batch size', default=1)

parser.add_argument('--gpu',metavar='GPU', type=str,
                    help='GPU id to use.', default="0")

parser.add_argument('--task',metavar='TASKID', type=str, default=wandb.run.name, 
                    help='Task id of this run.')

parser.add_argument('-n',metavar='N', type=int, default=80, 
                    help='Number of picture per patient.')

def create_dir_not_exist(path):
    for length in range(1, len(path.split(os.path.sep))):
        check_path = os.path.sep.join(path.split(os.path.sep)[:(length+1)])
        if not os.path.exists(check_path):
            os.mkdir(check_path)
            print(f'Created Dir: {check_path}')

def main():
    global args, best_prec1
    best_prec1 = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-6
    args.momentum  = 0.95
    args.decay  = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 100
    args.steps = [-1,1,100,150]
    args.scales = [1,1,1,1]
    args.workers = 32
    args.seed = time.time()
    args.print_freq = 30
    wandb.config.update(args)
    wandb.run.name = f"Default_{wandb.run.name}" if (args.task == wandb.run.name) else f"{args.task}_{wandb.run.name}"

    conf= configparser.ConfigParser()
    conf.read(args.config) 
    print(conf)
    TRAIN_DIR = conf.get("COVNet_CSR_Gate_raw", "train") 
    VALID_DIR = conf.get("COVNet_CSR_Gate_raw", "valid") 
    TEST_DIR = conf.get("COVNet_CSR_Gate_raw", "test") 
    LOG_DIR = conf.get("COVNet_CSR_Gate_he", "log") 
    create_dir_not_exist(LOG_DIR)
    train_list = [os.path.join(TRAIN_DIR, item) for item in os.listdir(TRAIN_DIR)]
    val_list = [os.path.join(VALID_DIR, item) for item in os.listdir(VALID_DIR)]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    model1 = COVNet_CSR_Gate(3)
    model1 = model1.cuda()
    model2 = COVNet_CSR_Gate(3)
    model2 = model2.cuda()
    criterion = nn.BCELoss(size_average = False).cuda()
    optimizer1 = torch.optim.Adam(model1.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay)
    optimizer2 = torch.optim.Adam(model2.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay)
    model1 = DataParallel_withLoss(model1, criterion)
    model2 = DataParallel_withLoss(model2, criterion)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)
        train(train_list, [model1,model2], criterion, [optimizer1,optimizer2], epoch)
        prec1 = validate(val_list, [model1,model2], criterion, epoch)
        with open(os.path.join(LOG_DIR, args.task + ".txt"), "a") as f:
            f.write("epoch " + str(epoch) + "  BCELoss: " +str(float(prec1)))
            f.write("\n")
        wandb.save(os.path.join(LOG_DIR, args.task + ".txt"))
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best BCELoss {BCELoss:.3f} '.format(BCELoss=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict1': model1.state_dict(),
            'state_dict2': model2.state_dict(),
            'best_prec1': best_prec1,
            'optimizer1' : optimizer1.state_dict(),
            'optimizer2' : optimizer2.state_dict(),
        }, is_best,args.task, epoch = epoch,path=os.path.join(LOG_DIR, args.task))


def train(train_list, models, criterion, optimizers, epoch):
    model1,model2=models[0],models[1]
    optimizer1,optimizer2=optimizers[0],optimizers[1]
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(dataset.listDataset(train_list, train = False, n=args.n), num_workers=args.workers, batch_size=args.batch_size, shuffle = True)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    model1.train()
    model2.train()
    end = time.time()
    for i,(img, target,gray_class)in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        img = img.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor).squeeze(1).cuda()
        # print(img.shape)
        # print(target.shape)
        if gray_class==0:
            loss,_ = model1(target, img)
            loss = loss.sum()
            losses.update(loss.item(), img.size(0))
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
        else:
            loss,_ = model2(target, img)
            loss = loss.sum()
            losses.update(loss.item(), img.size(0))
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
        
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
    wandb.log({'Train loss': losses.avg/args.batch_size})

def validate(val_list, models, criterion, epoch):
    print ('begin test')
    model1,model2=models[0],models[1]
    test_loader = torch.utils.data.DataLoader(dataset.listDataset(val_list,n=args.n), batch_size=args.batch_size, shuffle = False)
    model1.eval()
    model2.eval()
    BCELoss = 0
    with torch.no_grad():
        for i,(img, target,gray_class) in enumerate(test_loader):
            img = img.cuda()
            img = img.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor).squeeze(1).cuda()
            if gray_class==0:
                _, output = model1(target, img)
            else:
                _, output = model2(target, img)
            BCELoss += criterion(output.data, target)
    BCELoss = BCELoss/len(test_loader)/args.batch_size
    print(' * BCELoss {BCELoss:.3f} '.format(BCELoss=BCELoss))
    wandb.log({'epoch': epoch, 'Valid loss': BCELoss})
    return BCELoss


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
        #print(outputs.shape)
        loss = self.loss(outputs, targets)
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
