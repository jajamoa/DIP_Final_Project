import torch
import torch.nn as nn

import torchvision.models as models


def selfTransNet(pretrained_net=None,num_classes=3):
    model = models.densenet169(pretrained=True)
    if pretrained_net:
        pretrained_net = torch.load(pretrained_net)
        model.load_state_dict(pretrained_net)

    num_features = model.classifier.in_features
    model.classifier=nn.Linear(num_classes,3)

    return model