import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import os 

IM_SIZE = 256
BATCH_SIZE = 512
NUM_WORKERS = 4

def saveModel(epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':loss,
        }, 'model.pth')

def checkAcc(model, loader, dtype, device):
    model.eval()
    num_correct = 0

    for x,y in loader:
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        scores = model(x)
        conf, preds = torch.max(scores.data,1)
        num_correct += (preds == y).sum()               
    acc = float(num_correct)/len(loader.dataset)
    return acc
        

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean_std_transform = transforms.Compose([
        transforms.Resize(IM_SIZE),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0),
                             std = (1,1,1))])
    dataset = torchvision.datasets.ImageFolder(root='../data/mgrs',
                                               transform=mean_std_transform)
    if os.path.exists('mean_std.npy'):
        mean_std = np.load('mean_std.npy')
        total_mean = mean_std[0]
        total_std = mean_std[1]
        
    else:
        dataloader = DataLoader(dataset,
                                batch_size = BATCH_SIZE,
                                shuffle = False,
                                num_workers = NUM_WORKERS,
                                pin_memory = True)
        
        psum = torch.tensor([0.0,0.0,0.0])
        psum_sq = torch.tensor([0.0,0.0,0.0])
        for inputs in tqdm(dataloader):
            psum += inputs[0].sum(axis = [0,2,3])
            psum_sq += (inputs[0] ** 2).sum(axis = [0,2,3])
        count = len(dataset) * IM_SIZE * IM_SIZE
        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)
        print(total_mean)
        print(total_std)
        np.save('mean_std.npy',np.stack((total_mean,total_std)))
   
        
    
    train_transform = transforms.Compose([
        transforms.Resize(IM_SIZE),
        transforms.RandomCrop(IM_SIZE),
        transforms.RandomRotation(degrees=(-180,180)),
        transforms.RandomPerspective(distortion_scale=0.5,p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = total_mean,
            std = total_std)])
    vt_transform = transforms.Compose([
        transforms.Resize(IM_SIZE),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = total_mean,
            std = total_std)])
    
    train_idx, valid_idx, test_idx = torch.utils.data.random_split(dataset, [.7,.2,.1])
    train_idx = train_idx.indices
    valid_idx = valid_idx.indices
    test_idx = test_idx.indices
    train_dataset = torchvision.datasets.ImageFolder(root='../data/mgrs',
                                                     transform=train_transform)
    valid_test_dataset = torchvision.datasets.ImageFolder(root='../data/mgrs',
                                                        transform=vt_transform)
    train_dataset = torch.utils.data.Subset(train_dataset,train_idx)
    valid_dataset = torch.utils.data.Subset(valid_test_dataset,valid_idx)
    test_dataset = torch.utils.data.Subset(valid_test_dataset,test_idx)
    test_dataset = torchvision.datasets.ImageFolder(root='../test',
                                                    transform = vt_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size = BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory = True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory = True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory = True)
    
    # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(
                torch.nn.Conv2d(3,32,11),
                torch.nn.ReLU(inplace =True),
                torch.nn.MaxPool2d(kernel_size=2,stride=2),
                torch.nn.Conv2d(32,64,7),
                torch.nn.ReLU(inplace = True),
                torch.nn.MaxPool2d(kernel_size=2,stride=2),
                torch.nn.Conv2d(64,128,3),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2,stride=2),
                torch.nn.Conv2d(128,256,3),
                torch.nn.ReLU(inplace = True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(256,128,1),
                torch.nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d(output_size=(1,1)),
                torch.nn.Flatten(),
                torch.nn.Linear(in_features=128,out_features=63,bias=True)
                )
    # model = torch.nn.Sequential()
    # layer1 = torch.nn.Sequential(
    #     torch.nn.Conv2d(3,32,11),
    #     torch.nn.ReLU(inplace =True),
    #     torch.nn.MaxPool2d(kernel_size=2,stride=2)
    #     )
    # layer2 = torch.nn.Sequential(
    #     torch.nn.Conv2d(32,64,7),
    #     torch.nn.ReLU(inplace = True),
    #     torch.nn.MaxPool2d(kernel_size=2,stride=2))
    # layer3 = torch.nn.Sequential(
    #     torch.nn.Conv2d(64,128,3),
    #     torch.nn.ReLU(inplace=True),
    #     torch.nn.MaxPool2d(kernel_size=2,stride=2))     
    # layer4 = torch.nn.Sequential(
    #     torch.nn.Conv2d(128,256,3),
    #     torch.nn.ReLU(inplace = True),
    #     torch.nn.MaxPool2d(kernel_size=2, stride=2))
    # layer5 = torch.nn.Sequential(
    #     torch.nn.Conv2d(256,128,1),
    #     torch.nn.ReLU(inplace=True)
    #     )
                
    # pool = torch.nn.AdaptiveAvgPool2d(output_size=(1,1))
    # flatten = torch.nn.Flatten()
    # output = torch.nn.Linear(in_features=128,out_features=63,bias=True)
                
    # model.add_module('layer1',layer1)
    # model.add_module('layer2',layer2)
    # model.add_module('layer3',layer3)
    # model.add_module('layer4',layer4)
    # model.add_module('layer5',layer5)
    # model.add_module('pool', pool)
    # model.add_module('flatten',flatten)
    # model.add_module('output',output)    
    
    
    # model.fc = torch.nn.Linear(2048,63,bias=True)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = True
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
    # model.fc.weight.requires_grad = True
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    if os.path.exists('model.pth'):
        checkpoint = torch.load('model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    
    num_epochs = 100
    for epoch in range(epoch+1, num_epochs):
        print('Starting epoch %d / %d' % (epoch, num_epochs))
        model.train()
        cum_loss = 0
        for x,y in train_loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            loss = loss_fn(scores,y)
            cum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #train_acc = checkAcc(model, train_loader, dtype, device)
        #val_acc= checkAcc(model, valid_loader,dtype, device)
       #test_acc= checkAcc(model, test_loader,dtype, device)
        train_loss = cum_loss/(len(train_loader))
        print('loss:',train_loss)
        #print('Train acc ',train_acc, 'Train loss', train_loss)
        #print('val acc ', val_acc)
       # print('test acc ', test_acc)
           
        print()
        saveModel(epoch, model, optimizer, loss_fn)
    saveModel(epoch, model, optimizer, loss_fn)
    
    y_pred = []
    y_true = []
    model.eval()
    for inputs, labels in test_loader:
        output = model(inputs.cuda())
        output = torch.max(output.data,1)[1].cpu().numpy()
        y_pred.extend(output)
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)
    classes = dataset.classes
    cf_matrix = confusion_matrix(y_true,y_pred)
    cf_norm = cf_matrix.astype('float')/cf_matrix.sum(axis=1)[:,np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix = cf_norm,display_labels = classes)
    disp.plot()
    plt.imshow(cf_norm,cmap = 'hot')
    plt.xticks(ticks=range(len(classes)),rotation=50,labels=classes)
    plt.yticks(ticks=range(len(classes)),labels=classes)
    plt.colorbar()

    #plt.title('Landsat Focus Region Classification, 200 Epochs')

if __name__ == '__main__':
    main()