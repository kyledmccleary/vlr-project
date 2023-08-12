# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:09:42 2023

@author: kmccl
"""

import torch, torchvision
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

BATCH_SIZE = 16
NUM_WORKERS = 4

path = 'enb0_best_model.pt'
model_dict = torch.load(path)
model_state_dict = model_dict['model_state_dict']
model = EfficientNet.from_name('efficientnet-b0')
model._fc = torch.nn.Linear(1280,18, bias=True)
model.load_state_dict(model_state_dict)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

test_transform = T.Compose(
       [
           T.Resize(640),
           T.CenterCrop(640),
           T.ToTensor(),
           T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
       ]
   )

test_dataset = torchvision.datasets.ImageFolder('interest_ds/test',
                                                transform=test_transform)
test_dataloader = DataLoader(test_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)

def check_accuracy(model, loader, device):
    model.eval()
    num_correct = 0

    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        scores = model(x)
        conf, preds = torch.max(scores.data, 1)
        num_correct += (preds == y).sum()
    acc = float(num_correct)/len(loader.dataset)
    return acc



if __name__ == '__main__':
    acc = check_accuracy(model, test_dataloader, device)
    print(acc)