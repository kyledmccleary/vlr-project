import torch, torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils import data
import os
import rasterio
import numpy as np
from tqdm import tqdm
from getRandomSubImage import getRandomSubImage
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

KEY = '17R'
TARGET_WIDTH_M = 400000
TARGET_HEIGHT_M = 300000
IM_SIZE_MEANSTD = (2518,3172)
IM_SIZE_MEANSTD = (500)
DATA_PATH = '../../data3/modis3'
BATCH_SIZE = 32
NUM_WORKERS = 4
DISTORTION_SCALE = 0.5
P = 0.1
IM_SIZE = (288,384)
NUM_EPOCHS = 10
GSD = 1000
LR = 1e-3

CLASSES = ['17R','12R']
NUM_CLASSES = len(CLASSES)

SEED = 0

if SEED != None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

class DataLoaderKNET(data.Dataset):
    def __init__(self, folder_path, val_transform = None, train_transform = None, mean_std_transform = None):
        super(DataLoaderKNET, self).__init__()
        self.folder_path = folder_path
        folders = os.listdir(folder_path) 
        self.files = []
        self.labels = []
        for folder in folders:
            path = os.path.join(folder_path,folder)
            files = os.listdir(path)
            for file in files:
                if file.endswith('.png'):
                    self.files.append(file)
                    self.labels.append(folder)       
                
        self.val_transform = val_transform
        self.train_transform = train_transform
        self.mean_std_transform = mean_std_transform
        self.lonlat_file = 'lonlat.npy'
        
    def __getitem__(self,index):
        img_path = os.path.join(self.folder_path, self.labels[index], self.files[index])
        lonlat_path = os.path.join(self.folder_path, self.labels[index], self.lonlat_file)
        im = torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        lonlat_arr = np.load(lonlat_path)
        lonlat_arr = lonlat_arr.view((lonlat_arr.dtype[0],len(lonlat_arr.dtype.names)))
        if self.train_transform:
            if self.files[index].startswith('MODIS'):     
                im = im.numpy()
                im = np.transpose(im,(1,2,0))                          
                data, label = getRandomSubImage(im,lonlat_arr, self.labels[index], TARGET_WIDTH_M,TARGET_HEIGHT_M,GSD,segment = True)
                data = self.train_transform(data)
                box = torch.tensor(label).float()
        elif self.val_transform:
            if self.files[index].startswith('MODIS'):
                im = im.numpy()
                im = np.transpose(im,(1,2,0))  
                data, label = getRandomSubImage(im,lonlat_arr, self.labels[index], TARGET_WIDTH_M,TARGET_HEIGHT_M,GSD,segment = True)
                data = self.val_transform(data)
                box = torch.tensor(label).float()
        else:
            im = im.numpy()
            im = np.transpose(im,(1,2,0))  
            data = im
            data = self.mean_std_transform(data)
       
        label = self.labels[index]
        label = torch.tensor(CLASSES.index(label)).to(torch.int64)
        #label = torch.nn.functional.one_hot(label,num_classes=NUM_CLASSES)
        return data, label
    
    def __len__(self):
        return len(self.files)
  

def getMeanStd(device):
    if os.path.exists('siglonet_mean_std.npy'):
            mean_std = np.load('siglonet_mean_std.npy')
            total_mean = mean_std[0]
            total_std = mean_std[1]
    else:
        mean_std_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IM_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0,0,0),
                                 std = (1,1,1))
            ])          
        meanstd_dset = DataLoaderKNET(DATA_PATH, mean_std_transform = mean_std_transform)    
        dataloader = DataLoader(meanstd_dset,
                                batch_size = BATCH_SIZE,
                                shuffle = False,
                                num_workers = NUM_WORKERS,
                                pin_memory = True)
        psum = torch.tensor([0.0,0.0,0.0])
        psum_sq = torch.tensor([0.0,0.0,0.0])
        cnt = 0
        fst_moment = torch.empty(3)
        snd_moment = torch.empty(3)
        for inputs in tqdm(dataloader):
            x = inputs[0].to(device)
            b,c,h,w = x.shape
            nb_pixels = b*h*w
            psum = inputs[0].sum(axis = [0,2,3])
            psum_sq = (inputs[0] ** 2).sum(axis = [0,2,3])
            fst_moment = (cnt * fst_moment + psum) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + psum_sq) / (cnt + nb_pixels)
            cnt += nb_pixels
        total_mean = fst_moment.float()
        total_std = torch.sqrt(snd_moment - fst_moment**2).float()        
        np.save('siglonet_mean_std.npy',np.stack((total_mean,total_std)))
    print(total_mean)
    print(total_std)
    return total_mean, total_std

def getTrainValDsets(mean, std):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IM_SIZE),
        transforms.RandomPerspective(DISTORTION_SCALE, P),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)
        ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IM_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean = mean,
                             std = std)
        ])
    train_dset = DataLoaderKNET(DATA_PATH,train_transform= train_transform)
    train_idx, val_idx = torch.utils.data.random_split(train_dset,[0.8,0.2])
    train_idx = train_idx.indices
    val_idx = val_idx.indices
    val_dset = DataLoaderKNET(DATA_PATH, val_transform = val_transform)
    train_subset = torch.utils.data.Subset(train_dset, train_idx)
    val_subset = torch.utils.data.Subset(val_dset, val_idx)
    return train_dset, val_dset, train_subset, val_subset
    
class DownBlock(torch.nn.Module):    
    def __init__(self,in_channels,out_channels):
        super(DownBlock,self).__init__()
        self.down = torch.nn.Sequential()
        self.down.add_module('conv1',torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1, bias=False,padding='same'))
        # self.down.add_module('bn1',torch.nn.BatchNorm2d(num_features=out_channels,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True))
        # self.down.add_module('conv2', torch.nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1, bias=False,padding='same'))
        # self.down.add_module('bn2',torch.nn.BatchNorm2d(num_features=out_channels,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True))
        # self.down.add_module('conv3', torch.nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1, bias=False,padding='same'))
        self.down.add_module('bn3',torch.nn.BatchNorm2d(num_features=out_channels,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True))
        self.down.add_module('relu',torch.nn.ReLU(inplace=True))
        self.down.add_module('maxpool',torch.nn.MaxPool2d(kernel_size=2,stride=2))
    def forward(self,x):
        return self.down(x)
    
class UpBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels, k = None):
        super(UpBlock,self).__init__()
        mc = in_channels//2
        self.up = torch.nn.Sequential()
        self.up.add_module('conv1',torch.nn.Conv2d(in_channels=in_channels,out_channels=mc,kernel_size=3,stride=1,bias=False,padding='same'))
        # self.up.add_module('bn1',torch.nn.BatchNorm2d(num_features=mc,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True))
        # self.up.add_module('conv2',torch.nn.Conv2d(in_channels=mc,out_channels=mc,kernel_size=3,stride=1,bias=False,padding='same'))
        # self.up.add_module('bn2',torch.nn.BatchNorm2d(num_features=mc,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True))
        # self.up.add_module('conv3',torch.nn.Conv2d(in_channels=mc,out_channels=mc,kernel_size=3,stride=1,bias=False,padding='same'))
        self.up.add_module('bn3',torch.nn.BatchNorm2d(num_features=mc,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True))
        self.up.add_module('relu',torch.nn.ReLU(inplace=True))
        if k:
            self.up.add_module('conv4',torch.nn.Conv2d(in_channels=mc,out_channels=out_channels,kernel_size=k,stride=1))
        else:
            self.up.add_module('upconv1',torch.nn.ConvTranspose2d(in_channels=mc,out_channels=out_channels,kernel_size=2,stride=2))
    def forward(self,x):
        return self.up(x)

class InBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InBlock,self).__init__()
        self.inlayer = torch.nn.Sequential()
        self.inlayer.add_module('conv1',torch.nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding='same', bias=False))
        self.inlayer.add_module('bn1',torch.nn.BatchNorm2d(num_features=32,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True))
    def forward(self,x):
        return self.inlayer(x)

class ClassBlock(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(ClassBlock,self).__init__()
        self.classlayer = torch.nn.Sequential()
        self.classlayer.add_module('flatten',torch.nn.Flatten())
        self.classlayer.add_module('fc',torch.nn.Linear(in_features=in_features,out_features=out_features,bias=True))
    def forward(self,x):
        return self.classlayer(x)

class Model(torch.nn.Module):
    def __init__(self,in_channels,out_channels,feature_map=False):
        super(Model,self).__init__()
        self.inlayer = InBlock(in_channels,32)
        self.down1 = DownBlock(32,64)
        self.down2 = DownBlock(64,128)
        self.down3 = DownBlock(128,256)
        self.down4 = DownBlock(256,512)
        self.down5 = DownBlock(512,1024)
        self.up1 = torch.nn.Sequential()
        self.up1.add_module('upconv',torch.nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2))
        self.up2 = UpBlock(1024,256)
        self.up3 = UpBlock(512,128)
        self.up4 = UpBlock(256,64)
        self.up5 = UpBlock(128,32)
        self.up6 = UpBlock(64,1,k=1)
        self.classblock = ClassBlock(384*288,NUM_CLASSES)
        self.feature_map = feature_map
    def forward(self,x):
        out1 = self.inlayer(x)
        out2 = self.down1(out1)
        out3 = self.down2(out2)
        out4 = self.down3(out3)
        out5 = self.down4(out4)
        out6 = self.down5(out5)
        
        out7 = self.up1(out6)
        out8 = self.up2(torch.cat((out5,out7),dim=1))
        out9 = self.up3(torch.cat((out4,out8),dim=1))
        out10 = self.up4(torch.cat((out3,out9),dim=1))
        out11 = self.up5(torch.cat((out2,out10),dim=1))
        out12 = self.up6(torch.cat((out1,out11),dim=1))
        
        fm = torch.nn.ReLU()(out12)
        
        out = self.classblock(out12)
        
        if self.feature_map:
            return [out,fm]
        
        else:
            return out
                       
def getModel(device,feature_map = False):    
    model = Model(3,1,feature_map=feature_map)
    # model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    # model.fc = torch.nn.Linear(in_features=2048,out_features = 2,bias=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # for param in model.layer4.parameters():
    #     param.requires_grad = True
    # model.fc.weight.requires_grad = True
    model = model.to(device)
    return model
    
def getLossFn():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn

def getOptimizer(model,lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer

def loadModel(model, optimizer):
    if os.path.exists('siglonet_model.pth'):
        checkpoint = torch.load('siglonet_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    return epoch, model, optimizer

def train(model, loss_fn, optimizer, epoch, train_loader,val_loader, device):
    train_losses = []
    # train_mses = []
    # val_mses = []
    # train_abs_diffs = []
    # val_abs_diffs = []
    num_epochs = NUM_EPOCHS
    
    for epoch in range(epoch+1, num_epochs+1):
        print('Starting epoch %d / %d' % (epoch, num_epochs))
        model.train()
        cum_loss = 0
        for x,y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            loss = loss_fn(scores,y)
            cum_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = cum_loss/(len(train_loader))
        train_losses.append(train_loss)
        print('train loss:', train_loss)
        # if epoch % 5 == 0:
        #     train_mse, train_abs_diff = getMSE(model, train_loader, device)
        #     val_mse, val_abs_diff = getMSE(model, val_loader, device)
        #     train_mses.append(train_mse)
        #     val_mses.append(val_mse)
        #     train_abs_diffs.append(train_abs_diff)
        #     val_abs_diffs.append(val_abs_diff)
        #     print('train mse:', train_mse)
        #     print('train abs_diff:', train_abs_diff)
        #     print('val mse:', val_mse)
        #     print('val abs diff:', val_abs_diff)
        # print()
        saveModel(epoch, model, optimizer, loss_fn)
    train_losses = torch.tensor(train_losses).detach().cpu().numpy()
    # train_mses = torch.tensor(train_mses).detach().cpu().numpy()
    # val_mses = torch.tensor(val_mses).detach().cpu().numpy()
    np.save('train_loss', train_losses)
    # train_val_err = np.stack((train_mses,val_mses),axis=-1)
    # np.save('train_val_err',train_val_err)
    # train_abs_diffs = torch.stack(train_abs_diffs,axis=0).numpy()
    # val_abs_diffs = torch.stack(val_abs_diffs,axis=0).numpy()
    # np.save('train_abs_diffs',train_abs_diffs)
    # np.save('val_abs_diffs',val_abs_diffs)
        
def getMSE(model, loader, device):
    model.eval()
    total_diff_sq = 0
    abs_dif = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    for x,y in loader:
        x = x.to(device)
        scores = model(x).detach().cpu()
        x = x.detach().cpu()
        total_diff_sq += torch.sum((scores-y)**2)
        abs_dif += torch.sum(torch.abs(scores-y),axis=0)
    mse = total_diff_sq / len(loader.dataset)
    abs_dif = abs_dif / len(loader.dataset)
    return mse, abs_dif

def saveModel(epoch, model, optimizer, loss):
    torch.save({
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':loss,
        }, 'siglonet_model.pth')      

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    mean, std = getMeanStd(device)
    train_dset, val_dset, train_subset, val_subset = getTrainValDsets(mean,std)
    train_loader = DataLoader(train_subset, batch_size =BATCH_SIZE, shuffle=True,
                              num_workers = NUM_WORKERS, pin_memory = True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers = NUM_WORKERS, pin_memory =True)
    model = getModel(device,feature_map = False)
    loss_fn = getLossFn()
    optimizer = getOptimizer(model, LR)
    epoch, model, optimizer = loadModel(model, optimizer)
    
    train(model, loss_fn, optimizer, epoch, train_loader,val_loader, device)
            
if __name__ == '__main__':
    main()
        
        