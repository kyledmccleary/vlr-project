from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import csv

im = Image.open('sequence/ISSGT_00120.png')

im_arr = np.array(im)
im_arr = im_arr[:,:,:3]

model = YOLO('best.pt')
# results = model.add_callback(event, func)

import os

files = os.listdir('datasets/florida/test/images')
out_list = []
for i in range(len(files)):
    file = files[i]
    # im = Image.open(os.path.join('sequence2',file))
    im = cv2.imread(os.path.join('sequence2',file))
    im_arr = np.array(im)
    im_arr = im_arr[:,:,:3]
    h = im_arr.shape[1]
    w = im_arr.shape[0]
    scalew = 1024/w
    newh = int(h*scalew)
    im_arr = cv2.resize(im_arr,(newh,1024))
    
    out = model(im_arr)
    if len(out[0].boxes.boxes) >= 2:       
        out_list.append([i, out])


with open('italy_algo_boxes.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    l = 0
    label_dict = {}
    first = True
    for row in reader:
        if row:
            if first:
                first = False
                continue
            label_dict[str(l)] = row
            l+=1





to_file_list=[]
for i in range(len(out_list)):
    index = out_list[i][0]
    boxes = out_list[i][1][0].boxes.boxes
    im = cv2.imread(os.path.join('sequence2',files[i]))
    im_arr = np.array(im)
    im_arr = im_arr[:,:,:3]
    h = im_arr.shape[1]
    w = im_arr.shape[0]
    scalew = 1024/w
    newh = int(h*scalew)
    im_arr = cv2.resize(im_arr,(newh,1024))
    for box in boxes:
        x1,y1,x2,y2,score,label = box
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(im_arr,(x1,y1),(x2,y2),(0,255,0),3)
        cv2.imshow('im',im_arr)
        cv2.waitKey(0)
        print(box)   
        l = int(label)
        tl_lon, tl_lat, br_lon, br_lat = [float(x) for x in label_dict[str(l)]]
        center_lon = (tl_lon + br_lon)/2
        center_lat = (tl_lat + br_lat)/2
        out_info = [index,int((x2+x1)//2),int((y2+y1)//2),center_lon, center_lat, float(score),l]
        to_file_list.append(out_info)

with open('one_pass_output.csv','w') as csvfile:
    writer = csv.writer(csvfile,delimiter=',')
    header = ['frame# @1fps', 'x(pixel)','y(pixel)','x(longitude)','y(latitude)','score','label(lat/lon come from this)']
    writer.writerow(header)
    writer.writerows(to_file_list)