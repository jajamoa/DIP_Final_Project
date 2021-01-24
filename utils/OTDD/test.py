import sys

import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

from otdd import *
from plot import *
from otdd_pytorch import *

from mnist_helper import *

mnist_data = get_mnist_data()

usps_data = get_usps_data()

mnist_sample = mnist_data.subsample(5000, equal_classes=True)
usps_sample = usps_data.subsample(5000, equal_classes=True)

print(mnist_sample.features[0].shape)
# print(usps_data.classes)
sys.exit()

distance_tensorized = PytorchEuclideanDistance()
routine_tensorized = SinkhornTensorized(distance_tensorized)
cost_tensorized = SamplesLossTensorized(routine_tensorized)

outputs = cost_tensorized.distance(mnist_sample.features, usps_sample.features, return_transport=True)

print(outputs[0])

fig,_=plot_coupling(outputs[1].numpy(), outputs[2].numpy(), mnist_sample.labels.numpy(), usps_sample.labels.numpy(),
                  mnist_sample.classes, usps_sample.classes, figsize=(10,10), cmap='OrRd')
fig.savefig('FeatureDistance.png')

outputs2 = cost_tensorized.distance_with_labels(mnist_sample.features, usps_sample.features, 
                                              mnist_sample.labels, usps_sample.labels, gaussian_class_distance=False)

plot_coupling(outputs2[1].numpy(), 
              outputs2[2].numpy(), 
              labels.numpy(), 
              labels_he.numpy(),
                  classes, classes, figsize=(10,10), cmap='OrRd')

fig.savefig('LabelDistance.png')

fig,_=plot_class_distances(outputs2[3], classes, classes, 
                            cmap='OrRd', figsize=(10,8), text=True)

fig.savefig('ClassDistance.png')


