import torch, model
a = torch.empty(1, 20, 3, 512, 512)
FCN = model.COVNet_FCN(3)
FCN(a)