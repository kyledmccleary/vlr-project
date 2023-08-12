import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet
import os
from tqdm import tqdm

BATCH_SIZE = 16
NUM_WORKERS = 4


def save_model(epoch, model, optimizer, loss, best_test_f1, name=None):
    if name is None:
        name = 'multilabel_model.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_test_f1': best_test_f1
    }, name)


def check_f1(model, loss_fn, loader, device):
    model.eval()
    cum_tp = cum_fp = cum_fn = 0
    cum_loss = 0

    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        scores = model(x)
        loss = loss_fn(scores, y)
        cum_loss += loss.item()
        preds = (scores.data > 0.9) * 1
        tp, tn, fp, fn = get_tfpn(preds, y)
        cum_tp += tp
        cum_fp += fp
        cum_fn += fn
    F1 = (2*cum_tp) / (2*cum_tp + cum_fp + cum_fn)
    test_prec = cum_tp / (cum_tp + cum_fp)
    test_rec = cum_tp / (cum_tp + cum_fn)
    loss = float(cum_loss)/len(loader.dataset)
    return F1, loss, test_prec, test_rec


class MultiLabelDataset(Dataset):
    def __init__(self, base_dir, transform):
        self.base_dir = base_dir
        self.img_files = os.listdir(os.path.join(base_dir, 'images'))
        self.lab_files = os.listdir(os.path.join(base_dir, 'labels'))
        self.transform = transform
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, 'images', 
                                self.img_files[idx])
        image = Image.open(img_path)
        image = image.convert('RGB')
        image = self.transform(image)
        label = np.load(os.path.join(self.base_dir, 'labels',
                                     self.lab_files[idx]))
        label = torch.tensor(label, dtype=torch.float)
        return image, label


def main():
    # Define Transforms
    train_transform = T.Compose(
        [
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            T.RandomAffine(degrees=(-180, 180), translate=(0, 0.1), scale=(0.75, 1)),
            T.Resize(360),
            T.CenterCrop(480),
            T.ToTensor(),
            T.Normalize((0.09795914, 0.10673781, 0.11483832), 
                        (0.17475154, 0.16193452, 0.16501454))
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize(360),
            T.CenterCrop(480),
            T.ToTensor(),
            T.Normalize((0.09795914, 0.10673781, 0.11483832), 
                        (0.17475154, 0.16193452, 0.16501454))
        ]
    )

    train_dataset = MultiLabelDataset('./multi_label/train', train_transform)
    test_dataset = MultiLabelDataset('./multi_label/test', test_transform)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)

    test_dataloader = DataLoader(test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)


    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(1280, 16, bias=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    if os.path.exists('multilabel_model.pth'):
        checkpoint = torch.load('multilabel_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_test_f1 = checkpoint['best_test_f1']
    else:
        epoch = 0
        best_test_f1 = 0
        
    num_epochs = 262
    for epoch in range(epoch+1, num_epochs):
        print('Starting epoch %d / %d' % (epoch, num_epochs))
        model.train()
        cum_loss = 0
        cum_tp = cum_fp = cum_fn = 0
        for x, y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)
            cum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sigscores = torch.nn.Sigmoid()(scores)
            preds = (sigscores.data > 0.95) * 1
            tp, tn, fp, fn = get_tfpn(preds, y)
            cum_tp += tp
            cum_fp += fp
            cum_fn += fn
        # train_acc = float(num_correct) / len(train_dataloader.dataset)
        train_f1 = (2*cum_tp) / (2*cum_tp + cum_fp + cum_fn)
        train_loss = float(cum_loss) / len(train_dataloader.dataset)
        test_f1, test_loss, test_prec, test_rec = check_f1(model, loss_fn, test_dataloader, device)
        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            save_model(epoch, model, optimizer, loss_fn, best_test_f1,
                       name='best_multilabel_model.pth')
        print('Train F1 ', train_f1, 'Train loss', train_loss)
        print('Train Precision', cum_tp / (cum_tp + cum_fp))
        print('Train Recall', cum_tp / (cum_tp + cum_fn))
        print('Test F1', test_f1, 'Test loss', test_loss)
        print('Test precision', test_prec, 'test recall', test_rec)
        print('Best test F1', best_test_f1)
        print()
        save_model(epoch, model, optimizer, loss_fn, best_test_f1)
    save_model(epoch, model, optimizer, loss_fn, best_test_f1)


def get_tfpn(preds, label):
    tn = ((preds == 0) * (label == 0)).sum()
    fn = ((preds == 0) * (label == 1)).sum()
    fp = ((preds == 1) * (label == 0)).sum()
    tp = ((preds == 1) * (label == 1)).sum()
    return tp, tn, fp, fn

if __name__ == '__main__':
    main()