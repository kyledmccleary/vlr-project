import cv2
import torch, torchvision
from efficientnet_pytorch import EfficientNet
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
import numpy as np
import os

RC_MODEL_PATH = './models/RC.pt'
LD_MODEL_PRE = './models/'
LD_MODEL_SUF = '.pt'
LANDMARKS_PRE = './landmarks/'
LANDMARKS_SUF = '_outboxes.npy'
CLASS_DICT_PATH = 'class_dict.npy'

def im_pipe(image_path):
    im = Image.open(image_path).convert('RGB')
    assert im.width == 2592 and im.height == 1944
    proceed = check_zeros(im)
    if not proceed:
        return [], []
    region, conf = check_region(im)
    class_dict = np.load(CLASS_DICT_PATH, allow_pickle=True).item()
    class_dict_rev = dict((value, key) for key, value in class_dict.items())
    regions = [x for x in region]
    keys = [class_dict_rev[x] for x in regions]
    all_landmarks = []
    for key in keys:
        landmarks = landmark_detection(key, im)      
        lonlat_landmarks = class2lonlat(key, landmarks)
        all_landmarks += lonlat_landmarks
    return keys, all_landmarks

def class2lonlat(key, landmarks):
    path = LANDMARKS_PRE + key + LANDMARKS_SUF
    lds = np.load(path)
    ld_lons = (lds[:,0] + lds[:,2])/2
    ld_lats = (lds[:,1] + lds[:,3])/2
    lonlat_landmarks = []
    for landmark in landmarks:
        ld_cls = landmark[2]
        ld_lon = ld_lons[ld_cls]
        ld_lat = ld_lats[ld_cls]
        lonlat_landmarks.append([landmark[0], landmark[1], (ld_lon, ld_lat),
                                 landmark[3]])
    return lonlat_landmarks
    

def landmark_detection(key, im):
    model = YOLO(LD_MODEL_PRE + key + LD_MODEL_SUF)
    sections = section_image(im)
    results = model(sections)
    box_list = []
    i = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x,y,w,h = box.xywh[0]
            if i == 1:
                x = x + 1280
            if i == 2:
                y = y + 632
            if i == 3:
                x = x + 1280
                y = y + 632
            cl = box.cls[0]
            conf = box.conf[0]
            box_list.append([x,y,cl,conf])
        i += 1
    box_list = torch.tensor(box_list)
    out_list = []
    if len(box_list) > 0:
        for i in range(model.model.model[22].dfl.conv.in_channels):
            idxs = torch.where(box_list[:,2] == i)
            if len(idxs[0]) > 0:
                cl_i = box_list[idxs]
                idx = cl_i[:,2].argmax()
                x_px = round(cl_i[idx][0].item())
                y_px = round(cl_i[idx][1].item())
                cl = int(cl_i[idx][2].item())
                conf = cl_i[idx][3].item()
                out_list.append([x_px, y_px, cl, conf])
    return out_list
            
        
         
def check_region(im):
    print('Checking regions')
    model_path = RC_MODEL_PATH
    model_dict = torch.load(model_path)
    model_state_dict = model_dict['model_state_dict']
    model = EfficientNet.from_name('efficientnet-b0')
    model._fc = torch.nn.Linear(1280, 16, bias=True)
    model.load_state_dict(model_state_dict)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)  
    transform = T.Compose(
        [
            T.Resize(360),
            T.CenterCrop(480),
            T.ToTensor(),
            T.Normalize((0.09795914, 0.10673781, 0.11483832), 
                        (0.17475154, 0.16193452, 0.16501454))
        ]
    )
    x = transform(im).to(device)
    x.unsqueeze_(0)
    model.eval()
    scores = model(x)
    sigscores = torch.nn.Sigmoid()(scores)
    preds = (sigscores.data > 0.9) * 1
    pred = preds[0].cpu()
    pred = [x.item() for x in pred.nonzero()]
    conf = [sigscores[0][x].item() for x in pred]
    print('Regions found: ', pred)
    return pred, conf
    
def check_zeros(im):
    im_sum = np.sum(im, axis=2)
    if (im_sum == 0).sum() > (im.height*im.width)//10:
        return False
    else:
        return True

def section_image(im):
    s1 = im.crop((0,0,1312,1312))
    s2 = im.crop((1280,0,2592,1312))
    s3 = im.crop((0, 632, 1312, 1944))
    s4 = im.crop((1280, 632, 2592, 1944))
    return [s1,s2,s3,s4]







# base_path = './12R_orb_bing'
# frame_list = []
# # image_path = 'orb3 (98).png'
# image_paths = os.listdir(base_path)
# all_landmarks = []
# # nimp = []
# # for image_path in image_paths:
# #     if image_path.endswith('png'):
# #         val = image_path[6:-5]
# #         # val = image_path[:-4]
# #         newname = str(val).zfill(3)
# #         os.rename(os.path.join(base_path, image_path),
# #                   os.path.join(base_path, newname + '.png'))
# #         os.rename(os.path.join(base_path, image_path[:-4] + '.npy'),
# #                   os.path.join(base_path, newname + '.npy'))
# #         nimp.append(newname + '.png')
# for image_path in image_paths:
#     if image_path.endswith('png'):
#         keys, landmarks = run_pipeline(os.path.join(base_path,image_path))
#         im = cv2.imread(os.path.join(base_path,image_path))     
#         im_small = cv2.resize(im, (640,480))          
#         for landmark in landmarks:
#             x, y, lonlat, conf = landmark
#             x_sm = round(x/4.05)
#             y_sm = round(y/4.05)
#             lonlat_str = '(' + str(round(lonlat[0],4)) + ',' + str(round(lonlat[1],4)) + ')'
#             cv2.circle(im_small, (x_sm,y_sm), 5,
#                        [0,0,255], thickness=-1)
#             cv2.putText(im_small,
#                         lonlat_str + " " + str(round(conf,2)),
#                         (x_sm, y_sm), 0, 0.5, [0,0,255])
#         s = ''.join(str(key) + ' ' for key in keys)
#         cv2.putText(im_small,
#                     s,
#                     (320, 50), 0, 1, [255, 200, 255], thickness=2)
#         frame_list.append(im_small)
#         all_landmarks.append([image_path, landmarks])
# np.save('all_landmarks.npy', np.asanyarray(all_landmarks,dtype=object))
# vid = cv2.VideoWriter('vid_bing_3.mp4',cv2.VideoWriter_fourcc(*'mp4v'),
#                       2.0, (640,480))            
# for frame in frame_list:
#     vid.write(frame)
       
        
        