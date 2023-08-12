import torch, torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

import os 

import rasterio

from PIL import Image

import opts

IM_SIZE = opts.IM_SIZE
CLASSES = opts.CLASSES


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loadModel(path):
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Linear(2048,63,bias=True)
    checkpoint = torch.load('model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def readImage(path):
    if path[-3:] == 'tif' or path[-4:] == 'tiff':
        with rasterio.open(path) as src:
            arr = src.read()
            arr_int = np.uint8(arr)
            if arr_int.max() <= 1:
                arr = arr*255
            arr_rgb = np.uint8(np.transpose(arr,(1,2,0)))
            rgb = Image.fromarray(arr_rgb)
    else:
        img = Image.open(path)
        rgb = img.convert('RGB')
    
    out = transformImage(rgb)
    out = out.resize(1,out.shape[0],out.shape[1],out.shape[2])
    return out
        

def transformImage(rgb):
    mean_std = np.load('mean_std.npy')

    transform = transforms.Compose([
        transforms.Resize(IM_SIZE),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
           mean = mean_std[0],
           std = mean_std[1])])
    
    out = transform(rgb).float()
    return out

def predict(im, model):
    scores = model(im)
    conf, preds = torch.max(scores.data,1)
    med = scores.median()
    top2 = scores.topk(2)[0]
    top2_idx = scores.topk(2)[1]
    pred = CLASSES[preds[0]]
    return pred, [top2,top2_idx,med]

def makePrediction(filepath, model):
    im = readImage(filepath)
    pred, others = predict(im,model)
    return pred
    

# model = loadModel('model.pth')
# model.eval()

# path = '../test_saral'
# files = os.listdir(path)
# preds = []
# confs = []
# labels = []
# for file in files:
#     im = readImage(os.path.join(path,file))
#     im = im.resize(1,im.shape[0],im.shape[1],im.shape[2])
#     label = ''
#     for i in range(len(file)):
#         label += file[i]
#         if file[i+1] == '_':
#             break
#     # label = file[:2] + file[6]
#     pred, conf = predict(im,model)
#     preds.append(pred)
#     confs.append(conf)
#     labels.append(label)

# classes = []
# for i in range(len(preds)):
#     if not preds[i] in classes:
#         classes.append(preds[i])
#     if not labels[i] in classes:
#         classes.append(labels[i])
# cm = np.zeros((len(classes),len(classes)))
# for i in range(len(preds)):
#     cm[classes.index(preds[i]),classes.index(labels[i])] += 1

# cm = confusion_matrix(labels,preds,labels = classes )
# cm_display = ConfusionMatrixDisplay(cm,display_labels=classes)
# cm_display.plot()
# plt.show()