import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from torchvision import models, datasets, transforms, utils
from PIL import Image
import scipy.misc
from model import COVNet_CSR
import cv2

class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, targets, *inputs):
        outputs = self.model(*inputs)
#         loss = self.loss(outputs, torch.topk(targets.long(), 1)[1].squeeze(1))
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

 
if __name__ ==  '__main__':
#     savepath='./CSR_HIGH_Focalmodel_best_middle'
#     if not os.path.exists(savepath):
#         os.mkdir(savepath)
        
    PATH = '/home/mist/CSR_HE_Intepmodel_best.pth'
    
#     model = COVNet_CSR(3)
#     criterion = nn.CrossEntropyLoss().cuda()

#     model = DataParallel_withLoss(model, criterion)

#     trained_net = torch.load(PATH)
#     new_dict = {}
#     print(type(trained_net['state_dict']))
#     for k,v in trained_net['state_dict'].items():
#         new_dict[k.replace('module.','')]=v
#     model.load_state_dict(new_dict)
#     model = model.model
#     model = model.cuda()

    model = COVNet_CSR(3)
    criterion = nn.CrossEntropyLoss().cuda()

    model = DataParallel_withLoss(model, criterion)

    trained_net = torch.load(PATH)
    model.load_state_dict(trained_net['state_dict'])
    model = model.cuda()

    # pretrained_dict = resnet50.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # net.load_state_dict(model_dict)

    model.eval()
    img=cv2.imread('./CT0007.png')
    img=cv2.resize(img,(224,224));
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img=transform(img).cuda()
    img=img.unsqueeze(0)
    with torch.no_grad():
        start=time.time()
        out=model(None, img)
        print("total time:{}".format(time.time()-start))
        result=out.cpu().numpy()
        # ind=np.argmax(out.cpu().numpy())
        ind=np.argsort(result,axis=1)
        for i in range(5):
            print("predict:top {} = cls {} : score {}".format(i+1,ind[0,1000-i-1],result[0,1000-i-1]))
        print("done")