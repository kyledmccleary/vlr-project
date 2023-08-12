import torch, torchvision
from torchvision import transforms,datasets
from torch.utils.data.dataloader import DataLoader
from torch.utils import data
import os
import rasterio
import cv2
import numpy as np
import pyproj
from getMGRS import getMGRS
from tqdm import tqdm

TARGET_WIDTH_M = 400000
TARGET_HEIGHT_M = 300000
BATCH_SIZE = 64
NUM_WORKERS = 4
DISTORTION_SCALE = 0.5
P = 0.0
DATA_PATH = '.'
KEY = '17R'
IM_SIZE = 256
IM_WIDTH = 430

SEED = 0

if SEED != None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
class DataLoaderKNET(data.Dataset):
    def __init__(self, folder_path, val_transform= None, train_transform = None, mean_std_transform=None):
        super(DataLoaderKNET,self).__init__()
        self.folder_path = folder_path
        files = os.listdir(folder_path)
        self.files = []
        for file in files:
            if file.endswith('.tif'):
                self.files.append(file)
        self.val_transform = val_transform
        self.train_transform = train_transform
        self.mean_std_transform = mean_std_transform
        
    def __getitem__(self,index):
        img_path = os.path.join(self.folder_path,self.files[index])
        with rasterio.open(img_path) as raster:
            if self.train_transform:             
                data, label = rasterTransform(raster,rand_rot = True)
                data = self.train_transform(data)
                    
            elif self.val_transform:
                data,label = rasterTransform(raster,rand_rot = False)
                data = self.val_transform(data)
            elif self.mean_std_transform:
                data, label = rasterTransform(raster, rand_rot = False)
                data = self.mean_std_transform(data)
            else:
                data,label = rasterTransform(raster,rand_rot = False)
        label = torch.tensor(label)
        return data.float(), label.float()
    def __len__(self):
        return len(self.files)
        
def randomRotation(image,xs,ys):
    image_h = image.shape[0]
    image_w = image.shape[1]
    center_h = image_h//2
    center_w = image_w//2
    rotation = (np.random.rand() - 0.5)*180
    M = cv2.getRotationMatrix2D((center_w,center_h),rotation,1)
    rotated_image = cv2.warpAffine(image, M, (image_w,image_h))
    rotated_xs = cv2.warpAffine(xs, M, (image_w,image_h))
    rotated_ys = cv2.warpAffine(ys, M, (image_w,image_h))
    return rotated_image, rotated_xs, rotated_ys
 
def centerCrop(image,xs,ys,new_width,new_height):
    height = image.shape[0]
    width = image.shape[1]
    height_crop = height - new_height
    width_crop = width - new_width
    cropped_image = image[height_crop//2:height -height_crop//2,width_crop//2:width -width_crop//2]
    cropped_xs = xs[height_crop//2:height -height_crop//2,width_crop//2:width -width_crop//2]
    cropped_ys = ys[height_crop//2:height -height_crop//2,width_crop//2:width -width_crop//2]
    return cropped_image, cropped_xs, cropped_ys


def rasterTransform(raster,rand_rot):
    crs = raster.crs
    rgb = raster.read()
    rgbT = np.transpose(rgb,(1,2,0))
    width = raster.width
    height = raster.height
    left, bottom, right, top = raster.bounds
    width_m = right - left
    gsd = width_m/width
    
    new_width_m = TARGET_WIDTH_M
    new_height_m = TARGET_HEIGHT_M
    
    new_width = int(new_width_m / gsd)
    new_height = int(new_height_m / gsd)
    
    ####testing
    new_width = round(new_width,-1)
    new_height = round(new_height,-1)
    
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
    xs = np.array(xs)
    ys = np.array(ys)
    if rand_rot == True:
        rotated_image, rotated_xs, rotated_ys = randomRotation(rgbT,xs,ys)
    else:
        rotated_image, rotated_xs, rotated_ys = rgbT,xs,ys
    
    cropped_image, cropped_xs, cropped_ys = centerCrop(rotated_image,rotated_xs,rotated_ys,new_width,new_height)
    tl_x = cropped_xs[0,0]
    tl_y = cropped_ys[0,0]
    tr_x = cropped_xs[0,-1]
    tr_y = cropped_ys[0,-1]
    bl_x = cropped_xs[-1,0]
    bl_y = cropped_ys[-1,0]
    br_x = cropped_xs[-1,-1]
    br_y = cropped_ys[-1,-1]
    
    
    # print(crs, raster.name)
    # crs = 'EPSG:3857'
    transformer = pyproj.Transformer.from_crs(crs,'EPSG:4326',always_xy = True)
    top_left = transformer.transform(tl_x,tl_y)
    top_right = transformer.transform(tr_x,tr_y)
    bot_left = transformer.transform(bl_x,bl_y)
    bot_right = transformer.transform(br_x,br_y)
    
    grid = getMGRS() 
    grid_left, grid_bottom, grid_right, grid_top = grid[KEY]   
    grid_left = grid_left - 6
    grid_bottom = grid_bottom - 8
    grid_right = grid_right + 6
    grid_top = grid_top + 8
    
    top_left_x = (top_left[0] -grid_left)/(grid_right-grid_left)
    top_left_y = (top_left[1] - grid_bottom )/(grid_top - grid_bottom)
    top_right_x = (top_right[0] -grid_left)/(grid_right-grid_left)
    top_right_y = (top_right[1] -grid_bottom)/(grid_top - grid_bottom)
    bot_right_x = (bot_right[0] -grid_left)/(grid_right-grid_left)
    bot_right_y = (bot_right[1] -grid_bottom)/(grid_top - grid_bottom)
    bot_left_x = (bot_left[0] -grid_left)/(grid_right-grid_left)
    bot_left_y = (bot_left[1] -grid_bottom)/(grid_top - grid_bottom)
    
    # top_left_x = (top_left[0] + 180)/360
    # top_left_y = (top_left[1] + 90)/180
    # top_right_x = (top_right[0] + 180)/360
    # top_right_y = (top_right[1] + 90)/180
    # bot_right_x = (bot_right[0] + 180)/360
    # bot_right_y = (bot_right[1] + 90)/180
    # bot_left_x = (bot_left[0] + 180)/360
    # bot_left_y = (bot_left[1] + 90)/180
    
    label = (top_left_x,top_left_y,top_right_x,top_right_y,bot_left_x,bot_left_y,bot_right_x,bot_right_y)
    data = cropped_image
    return data, label
 
def getMSE(model, loader, device):
    model.eval()
    total_diff_sq = 0
    abs_dif = torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    for x,y in loader:
        x = x.to(device)
        scores = model(x)
        x = x.detach().cpu()
        scores = scores.detach().cpu()
        total_diff_sq += torch.sum((scores-y)**2)
        abs_dif += torch.sum(torch.abs(scores-y),axis=0)
    mse = total_diff_sq / len(loader.dataset)
    abs_dif = abs_dif / len(loader.dataset)
    return mse, abs_dif
    
def saveModel(epoch, model, optimizer, loss):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':loss,
        }, 'model.pth')
   
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean_std_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IM_SIZE)),
      #  transforms.CenterCrop((IM_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0,0,0),
                             std = (1,1,1))])
    dset = DataLoaderKNET(DATA_PATH,mean_std_transform=mean_std_transform)
    if os.path.exists('mean_std.npy'):
        mean_std = np.load('mean_std.npy')
        total_mean = mean_std[0]
        total_std = mean_std[1]
        
    else:
        dataloader = DataLoader(dset,
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
            b,c,h,w = inputs[0].shape
            nb_pixels = b*h*w
            psum = inputs[0].sum(axis = [0,2,3])
            psum_sq = (inputs[0] ** 2).sum(axis = [0,2,3])
            fst_moment = (cnt * fst_moment + psum) / (cnt + nb_pixels)
            snd_moment = (cnt * snd_moment + psum_sq) / (cnt + nb_pixels)
            cnt += nb_pixels
        #count = len(dset) * IM_SIZE * IM_SIZE
       # print(count)
        # total_mean = psum / count
        # print(total_mean)
        # total_var = (psum_sq / count) - (total_mean ** 2)
        # print(total_var)
        # total_std = torch.sqrt(total_var)
        total_mean = fst_moment.float()
        total_std = torch.sqrt(snd_moment - fst_moment**2).float()
        print(total_mean)
        print(total_std)
        np.save('mean_std.npy',np.stack((total_mean,total_std)))

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((600,800)),
        # transforms.CenterCrop((600,800)),
        transforms.Resize((IM_SIZE)),
      #  transforms.CenterCrop((IM_SIZE,IM_WIDTH)),
        transforms.RandomPerspective(DISTORTION_SCALE, P),
        transforms.ToTensor(),
        transforms.Normalize(mean = total_mean,
                             std = total_std)
        ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((600,800)),
        # transforms.CenterCrop((600,800)),
        transforms.Resize((IM_SIZE)),
     #   transforms.CenterCrop((IM_SIZE,IM_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean = total_mean,
                             std = total_std)
        ])
    
    dset = DataLoaderKNET(DATA_PATH)
    train_idx, val_idx = torch.utils.data.random_split(dset, [0.8,0.2])
    train_idx = train_idx.indices
    val_idx = val_idx.indices
    train_dset = DataLoaderKNET(DATA_PATH,train_transform=train_transform)
    val_dset = DataLoaderKNET(DATA_PATH,val_transform=val_transform)
    train_subset = torch.utils.data.Subset(train_dset,train_idx)
    val_subset = torch.utils.data.Subset(val_dset,val_idx)
    train_loader = DataLoader(train_subset, batch_size = BATCH_SIZE, shuffle=True,
                              num_workers = NUM_WORKERS, pin_memory = True)
    val_loader = DataLoader(val_subset,batch_size = BATCH_SIZE, shuffle=False,
                            num_workers = NUM_WORKERS, pin_memory = True)
    
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048,8,bias=True)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc.weight.requires_grad = True
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    if os.path.exists('model.pth'):
        checkpoint = torch.load('model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    
    train_losses = []
    train_mses = []
    val_mses = []
    train_abs_diffs = []
    val_abs_diffs = []
    
    num_epochs = 50
    for epoch in range(epoch+1, num_epochs+1):
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
        train_mse, train_abs_diff = getMSE(model, train_loader, device)
        val_mse, val_abs_diff = getMSE(model, val_loader, device)
       #test_acc= checkAcc(model, test_loader,dtype, device)
        train_loss = cum_loss/(len(train_loader))
        train_losses.append(train_loss)
        train_mses.append(train_mse)
        val_mses.append(val_mse)
        train_abs_diffs.append(train_abs_diff)
        val_abs_diffs.append(val_abs_diff)
        
        
        print('train mse:',train_mse)
        print('train loss:',train_loss)
        print('val_mse:',val_mse)
        print('train abs diffs:',train_abs_diff)
        print('val abs diffs:', val_abs_diff)
        
        #print('Train acc ',train_acc, 'Train loss', train_loss)
        #print('val acc ', val_acc)
       # print('test acc ', test_acc)
           
        print()
        saveModel(epoch, model, optimizer, loss_fn)
    train_losses = torch.tensor(train_losses).detach().cpu().numpy()
    train_mses = torch.tensor(train_mses).detach().cpu().numpy()
    val_mses = torch.tensor(val_mses).detach().cpu().numpy()
    train_val_loss_err = np.stack((train_losses,train_mses,val_mses),axis=-1)
    print(train_val_loss_err,train_val_loss_err.shape)
    np.save('train_val_loss_err',train_val_loss_err)
    train_abs_diffs = torch.stack(train_abs_diffs,axis=0).numpy()
    print(train_abs_diffs,train_abs_diffs.shape)
    val_abs_diffs = torch.stack(val_abs_diffs,axis=0).numpy()
    np.save('train_abs_diffs',train_abs_diffs)
    np.save('val_abs_diffs',val_abs_diffs)

if __name__ == '__main__':
    main()
    