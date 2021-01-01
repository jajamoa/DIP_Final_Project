import sys
import os
import warnings
from moco import MoCo
from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import torch.backends.cudnn as cudnn
import torch.distributed as dist

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

parser.add_argument('--selftrans', '-self', metavar='SELFTRANS', default="Self-Trans.pt", type=str,
                    help='path to the self-trans model')

parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE' ,type=int,
                    help='batch size', default=2)

parser.add_argument('--gpu',metavar='GPU', type=str,
                    help='GPU id to use.', default="0")

parser.add_argument('--task',metavar='TASKID', type=str, default=wandb.run.name, 
                    help='Task id of this run.')


# moco specific configs:
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=128, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', default=True,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=True,
                    help='use cosine lr schedule')

# options for distributed training
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')


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
    args.epochs = 1
    args.steps = [-1,1,100,150]
    args.scales = [1,1,1,1]
    args.workers = 0
    args.seed = time.time()
    args.print_freq = 30
    wandb.config.update(args)
    wandb.run.name = f"Default_{wandb.run.name}" if (args.task == wandb.run.name) else f"{args.task}_{wandb.run.name}"

    conf= configparser.ConfigParser()

    conf.read(args.config) 
    TRAIN_DIR = conf.get("moco", "train") 
    VALID_DIR = conf.get("moco", "valid") 
    TEST_DIR = conf.get("moco", "test") 
    LOG_DIR = conf.get("moco", "log") 
    create_dir_not_exist(LOG_DIR)
    train_list = [os.path.join(TRAIN_DIR, item) for item in os.listdir(TRAIN_DIR)]
    train_list=checkSuffix(train_list)
    val_list = [os.path.join(VALID_DIR, item) for item in os.listdir(VALID_DIR)]
    val_list=checkSuffix(val_list)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)

    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                        world_size=args.world_size, rank=args.rank)

    
    model = MoCo(args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay)
    model = DataParallel_withLoss(model, criterion)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion, epoch)
        with open(os.path.join(LOG_DIR, args.task + ".txt"), "a") as f:
            f.write("epoch " + str(epoch) + "  CELoss: " +str(float(prec1)))
            f.write("\n")
        wandb.save(os.path.join(LOG_DIR, args.task + ".txt"))
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best CELoss {CELoss:.5f} '.format(CELoss=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task, epoch = epoch,path=os.path.join(LOG_DIR, args.task))


def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_loader = torch.utils.data.DataLoader(dataset.mocoDataset(train_list,aug_plus=args.aug_plus), 
    num_workers=args.workers, batch_size=args.batch_size, shuffle = True)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    

    
    model.train()
    end = time.time()
    for i,(images, _)in enumerate(train_loader):
        data_time.update(time.time() - end)
        # img = img.cuda()
        # img = img.type(torch.FloatTensor)
        # print(img.shape)
        loss, _ = model(images)
        loss = loss.sum()
        losses.update(loss.item(), images.size(2))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {lossval:.6f} ({lossavg:.6f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, lossval=losses.val/args.batch_size, lossavg=losses.avg/args.batch_size))
    wandb.log({'CEloss': losses.avg/args.batch_size})

def validate(val_list, model, criterion, epoch):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(dataset.mocoDataset(val_list,aug_plus=args.aug_plus),
                                              batch_size=args.batch_size, shuffle = False)
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')

    model.eval()
    CELoss = 0
    with torch.no_grad():
        for i,(images, target) in enumerate(test_loader):
            # img = img.cuda()
            # img = img.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor).squeeze(1).cuda()
            _, output  = model(images)

            CELoss += criterion(output.data,torch.topk(target.long(), 1)[1].squeeze(1))

            # acc1/acc5 are (K+1)-way contrast classifier accuracy
            # measure accuracy
            # acc1, acc5 = accuracy(output, target, topk=(1, 5))
            # top1.update(acc1[0], images[0].size(0))
            # top5.update(acc5[0], images[0].size(0))

    CELoss = CELoss/len(test_loader)/args.batch_size
    # ACC1 = top1.avg/args.batch_size
    # ACC5 = top5.avg/args.batch_size

    print(' * CELoss {CELoss:.5f} '.format(CELoss=CELoss))
    # print(' * ACC1 {ACC1:.3f} '.format(ACC1=ACC1))
    # print(' * ACC5 {ACC5:.3f} '.format(ACC5=ACC5))

    wandb.log({'epoch': epoch, 'CEloss': CELoss})
    # wandb.log({'epoch': epoch, 'ACC1': ACC1})
    # wandb.log({'epoch': epoch, 'ACC5': ACC5})

    return CELoss


def adjust_learning_rate(optimizer, epoch):
    """Decay the learning rate based on schedule"""
    args.lr = args.original_lr
    if args.cos:  # cosine lr schedule
        args.lr *= 0.5 * (1. + np.cos(np.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            args.lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
    wandb.log({'Learning Rate': args.lr})

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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

    def forward(self,*inputs):
        outputs, target = self.model(*inputs)
        loss = self.loss(outputs, target)
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
