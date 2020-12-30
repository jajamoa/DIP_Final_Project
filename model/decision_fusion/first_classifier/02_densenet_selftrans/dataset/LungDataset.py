import h5py 
import numpy as np
import skimage
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile
from PIL import Image


class LungDataset(Dataset):
    def __init__(self, img, label, transform=None):
        self.img = img
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = PIL_image = Image.fromarray(self.img[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.label[idx])}
        return sample

if __name__=='__main__':
    normalize = transforms.Normalize(mean=[0.45271412, 0.45271412, 0.45271412],
                                        std=[0.33165374, 0.33165374, 0.33165374])
    train_transformer = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(90),
        # random brightness and random contrast
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize
    ])

    val_transformer = transforms.Compose([
    #     transforms.Resize(224),
    #     transforms.CenterCrop(224),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])



    f = h5py.File('all_patches.hdf5','r')
    f.keys()
    img = f['ct_slices'][:]  
    label = f['slice_class'][:] 
    f.close()
    print(np.shape(img))
    print('b',np.shape(label))
    skimage.io.imshow(img[120])
    print(label[120])
    batchsize=4
    
    trainset = LungDataset(img, label, transform= val_transformer)
    valset = LungDataset(img, label, transform= val_transformer)
    train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)
    modelname = 'medical_transfer'