import torch, model
a = torch.empty(1, 2, 3, 512, 512)
FCN = model.COVNet_CBAM(3)
FCN(a)


