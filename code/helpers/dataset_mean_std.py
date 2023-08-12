# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 19:29:05 2023

@author: kmccl
"""

import torch, torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

BATCH_SIZE = 64
NUM_WORKERS = 0

class TrainDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.transform = transform
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.files[idx])
        image = Image.open(img_path)
        image = image.convert('RGB')
        image = self.transform(image)
        return image, 0

def get_mean_std(loader):
    # VAR[X] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
    mean = channels_sum / num_batches
    square_mean = channels_squared_sum / num_batches
    std = (square_mean - mean**2)**(0.5)
    return mean, std


if __name__ == '__main__':
    path = './multi_label/train/images'
    transform = T.Compose([
        T.Resize(360),
        T.CenterCrop(480),
        T.ToTensor()])
    dset = TrainDataset(path, transform)
    loader = DataLoader(dset,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS)
    
    mean, std = get_mean_std(loader)
    
    out = np.array([mean, std])
    
    np.save('mean_std_multi_label.npy', out)