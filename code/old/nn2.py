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
DATA_PATH = '../../data3/modis2/17R'
BATCH_SIZE = 512
NUM_WORKERS = 4
DISTORTION_SCALE = 0.1
P = 0.1
IM_SIZE = (288,384)
NUM_EPOCHS = 1000
GSD = 1000
LR = 1e-3

class DataLoaderKNET(data.Dataset):
    def __init__(self, folder_path, val_transform = None, train_transform = None, mean_std_transform = None):
        super(DataLoaderKNET, self).__init__()
        self.folder_path = folder_path
        files = os.listdir(folder_path)
        self.files = []
        for file in files:
            if file.endswith('.npy') and not file.startswith(KEY):
                self.files.append(file)
                
        self.val_transform = val_transform
        self.train_transform = train_transform
        self.mean_std_transform = mean_std_transform
        self.lonlat_file = KEY + '_lonlat.npy'
    
    def __getitem__(self,index):
        img_path = os.path.join(self.folder_path, self.files[index])
        lonlat_path = os.path.join(self.folder_path, self.lonlat_file)
        im = np.load(img_path)
        im = im.view((im.dtype[0],len(im.dtype.names)))
        lonlat_arr = np.load(lonlat_path)
        lonlat_arr = lonlat_arr.view((lonlat_arr.dtype[0],len(lonlat_arr.dtype.names)))
        if self.train_transform:
            if self.files[index].startswith('MODIS'):                              
                data, label = getRandomSubImage(im,lonlat_arr, KEY, TARGET_WIDTH_M,TARGET_HEIGHT_M,GSD,segment = False)
                data = self.train_transform(data)
                label = torch.tensor(label).float()
        elif self.val_transform:
            if self.files[index].startswith('MODIS'):
                data, label = getRandomSubImage(im,lonlat_arr, KEY, TARGET_WIDTH_M,TARGET_HEIGHT_M,GSD,segment = False)
                data = self.val_transform(data)
                label = torch.tensor(label).float()
        else:
            data = im
            label = 1
            data = self.mean_std_transform(data)
       
        return data, label
    
    def __len__(self):
        return len(self.files)
  

def getMeanStd(device):
    if os.path.exists('mean_std.npy'):
            mean_std = np.load('mean_std.npy')
            total_mean = mean_std[0]
            total_std = mean_std[1]
    else:
        mean_std_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IM_SIZE_MEANSTD),
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
        np.save('mean_std.npy',np.stack((total_mean,total_std)))
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
    
def getModel(device):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048,8,bias=True)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc.weight.requires_grad = True
    
    # model = torchvision.models.resnet50(weights=None)
    # model.fc = torch.nn.Linear(2048,8,bias=True)
    # model = model.to(device)
    # for param in model.parameters():
    #     param.requires_grad = True
    
    # model = torchvision.models.segmentation.fcn_resnet50(weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT)
    
    # model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    return model
    
def getLossFn():
    loss_fn = torch.nn.MSELoss()
    return loss_fn

def getOptimizer(model,lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer

def loadModel(model, optimizer):
    if os.path.exists('model.pth'):
        checkpoint = torch.load('model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    return epoch, model, optimizer

def train(model, loss_fn, optimizer, epoch, train_loader,val_loader, device):
    train_losses = []
    train_mses = []
    val_mses = []
    train_abs_diffs = []
    val_abs_diffs = []
    num_epochs = NUM_EPOCHS
    
    for epoch in tqdm(range(epoch+1, num_epochs+1)):
        print('Starting epoch %d / %d' % (epoch, num_epochs))
        model.train()
        cum_loss = 0
        for x,y in train_loader:
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
        if epoch % 5 == 0:
            train_mse, train_abs_diff = getMSE(model, train_loader, device)
            val_mse, val_abs_diff = getMSE(model, val_loader, device)
            train_mses.append(train_mse)
            val_mses.append(val_mse)
            train_abs_diffs.append(train_abs_diff)
            val_abs_diffs.append(val_abs_diff)
            print('train mse:', train_mse)
            print('train abs_diff:', train_abs_diff)
            print('val mse:', val_mse)
            print('val abs diff:', val_abs_diff)
            np.save('train_loss'+str(epoch),torch.tensor(train_losses).detach().cpu().numpy())
            np.save('train_val_err' + str(epoch), np.stack((torch.tensor(train_mses).detach().cpu().numpy(), torch.tensor(val_mses).detach().cpu().numpy()),axis=-1))
            np.save('train_abs_diffs' + str(epoch),torch.stack(train_abs_diffs,axis=0).numpy())
            np.save('val_abs_diffs' + str(epoch), torch.stack(val_abs_diffs,axis=0).numpy())
            
        print()
            
        saveModel(epoch, model, optimizer, loss_fn)
    train_losses = torch.tensor(train_losses).detach().cpu().numpy()
    train_mses = torch.tensor(train_mses).detach().cpu().numpy()
    val_mses = torch.tensor(val_mses).detach().cpu().numpy()
    np.save('train_loss', train_losses)
    train_val_err = np.stack((train_mses,val_mses),axis=-1)
    np.save('train_val_err',train_val_err)
    train_abs_diffs = torch.stack(train_abs_diffs,axis=0).numpy()
    val_abs_diffs = torch.stack(val_abs_diffs,axis=0).numpy()
    np.save('train_abs_diffs',train_abs_diffs)
    np.save('val_abs_diffs',val_abs_diffs)
        
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
        }, 'model.pth')      

       

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    mean, std = getMeanStd(device)
    train_dset, val_dset, train_subset, val_subset = getTrainValDsets(mean,std)
    train_loader = DataLoader(train_subset, batch_size =BATCH_SIZE, shuffle=True,
                              num_workers = NUM_WORKERS, pin_memory = True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers = NUM_WORKERS, pin_memory =True)
    model = getModel(device)
    loss_fn = getLossFn()
    optimizer = getOptimizer(model, LR)
    epoch, model, optimizer = loadModel(model, optimizer)
    
    train(model, loss_fn, optimizer, epoch, train_loader,val_loader, device)
    
    
    
    
if __name__ == '__main__':
    main()
        
        