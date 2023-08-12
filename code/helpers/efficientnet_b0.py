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

# orig_img = Image.open('l8_17R_00014.tif')
# torch.manual_seed(0)

BATCH_SIZE = 16
NUM_WORKERS = 4

# def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
#     if not isinstance(imgs[0], list):
#         imgs = [imgs]
#
#     num_rows = len(imgs)
#     num_cols = len(imgs[0]) + with_orig
#     fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
#     for row_idx, row in enumerate(imgs):
#         row = [orig_img] + row if with_orig else row
#         for col_idx, img in enumerate(row):
#             ax = axs[row_idx, col_idx]
#             ax.imshow(np.asarray(img), **imshow_kwargs)
#             ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
#
#     if with_orig:
#         axs[0, 0].set(title='Original Image')
#         axs[0, 0].title.set_size(8)
#     if row_title is not None:
#         for row_idx in range(num_rows):
#             axs[row_idx, 0].set(ylabel=row_title[row_idx])
#
#     plt.tight_layout()
#     plt.show()

def save_model(epoch, model, optimizer, loss, best_val_acc, name=None):
    if name is None:
        name = 'model.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_val_acc': best_val_acc
    }, name)


def check_accuracy(model, loss_fn, loader, device):
    model.eval()
    num_correct = 0
    cum_loss = 0

    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)
        scores = model(x)
        loss = loss_fn(scores, y)
        cum_loss += loss.item()
        conf, preds = torch.max(scores.data, 1)
        num_correct += (preds == y).sum()
    acc = float(num_correct)/len(loader.dataset)
    loss = float(cum_loss)/len(loader.dataset)
    return acc, loss


def main():
    # Define Transforms
    train_transform = T.Compose(
        [
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            T.RandomAffine(degrees=(-180, 180), translate=(0, 0.1), scale=(0.75, 1)),
            T.Resize(480),
            T.CenterCrop(480),
            T.ToTensor(),
            T.Normalize((0.09795914, 0.10673781, 0.11483832),
                        (0.17475154, 0.16193452, 0.16501454))
        ]
    )

    val_transform = T.Compose(
        [
            T.Resize(480),
            T.CenterCrop(480),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder('interest_ds/train', transform=train_transform)
    val_dataset = torchvision.datasets.ImageFolder('interest_ds/test', transform=val_transform)
    
    #train_set, val_set = torch.utils.data.random_split(train_dataset, [round(.8*len(train_dataset)),round(.2*len(train_dataset))])
    #val_set = torch.utils.data.Subset(val_dataset,val_set.indices)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS,
                                  pin_memory=True)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)



    # fig, axs = plt.subplots(nrows=5, ncols=20)
    # for i in range(5):
    #     for j in range(20):
    #         transformed_imgs = train_transform(orig_img)
    #         axs[i,j].imshow(transformed_imgs)
    # plt.show()
    # plot(transformed_imgs, with_orig=False)

    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(1280, 18, bias=True)
    print(model)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = True

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists('model.pth'):
        checkpoint = torch.load('model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
    else:
        epoch = 0
        best_val_acc = 0

    num_epochs = 1000
    for epoch in range(epoch+1, num_epochs):
        print('Starting epoch %d / %d' % (epoch, num_epochs))
        model.train()
        cum_loss = 0
        num_correct = 0
        for x, y in tqdm(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)
            cum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            conf, preds = torch.max(scores.data, 1)
            num_correct += (preds == y).sum()
        train_acc = float(num_correct) / len(train_dataloader.dataset)
        train_loss = float(cum_loss) / len(train_dataloader.dataset)
        val_acc, val_loss = check_accuracy(model, loss_fn, val_dataloader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(epoch, model, optimizer, loss_fn, best_val_acc, name='best_model.pth')
        print('Train acc ', train_acc, 'Train loss', train_loss)
        print('Val acc', val_acc, 'Val loss', val_loss)
        print('Best val acc', best_val_acc)
        print()
        save_model(epoch, model, optimizer, loss_fn, best_val_acc)
    save_model(epoch, model, optimizer, loss_fn, best_val_acc)


if __name__ == '__main__':
    main()