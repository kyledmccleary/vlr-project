import numpy as np
import cv2
import os
import csv

def rotateImage(image,rotation):
    image_h = image.shape[0]
    image_w = image.shape[1]
    center_h = image_h//2
    center_w = image_w//2
    M = cv2.getRotationMatrix2D((center_w,center_h),rotation,1)
    rotatedImage = cv2.warpAffine(image, M,(image_w,image_h))
    return rotatedImage, M

folder_path = 'l8bboxtest'
data_path = 'data'
thumb_path = 'thumbs'
label_path = 'labels'

files = os.listdir(os.path.join(folder_path,data_path))
# files = [files[1]]
for file in files:  
    polygons_path = 'polygons'
    
    polygon_path = os.path.join(folder_path,polygons_path,file[:-4]+'.csv')
    
    im = cv2.imread(os.path.join(folder_path,data_path,file))
    
    polys = []
    labels = []
    if os.path.exists(polygon_path):
        with open(polygon_path) as csvfile:
            reader = csv.reader(csvfile, delimiter=',')            
            for row in reader:
                print(row)
                first = True
                poly = []
                if not(row):
                    continue       
                for block in row:
                    if first:
                        label = int(block)
                        labels.append(label)
                        first = False
                    else:
                        xy = ''
                        for char in block:
                            if char in ['(',')', ',']:
                                continue
                            else:
                                xy += char
                        xy = xy.split()
                        x = int(xy[0])
                        y = int(xy[1])
                        poly.append((x,y))
                polys.append(poly)
    
    rotations = [-135,-90,-45,45,90,135,180]
  
    for rotation in rotations:
        im_copy = im.copy()
        rotated,M = rotateImage(im_copy,rotation)
        out_name = file[:-4] + '_'+str(rotation)+'.png'
        out_path = os.path.join(folder_path,data_path,out_name)
        cv2.imwrite(out_path,rotated)
        rects = []
        if polys:
            for poly in polys:
                    poly_ray = np.array([[coord] for coord in poly])
                    new_poly = cv2.transform(poly_ray,M)
                    x,y,w,h = cv2.boundingRect(new_poly)
                    rotated = cv2.rectangle(rotated,(x,y),(x+w,y+h),color=[0,0,255],thickness=2)
                    rects.append([x,y,w,h])
        rotated = cv2.resize(rotated,(rotated.shape[1]//2,rotated.shape[0]//2))
        out_thumb = os.path.join(folder_path,thumb_path,out_name)
        cv2.imwrite(out_thumb,rotated)
    
        out_labels=[]
        for i in range(len(rects)): 
            x,y,w,h = rects[i]           
            out_labels.append([labels[i], x, y, w, h])   
            
        with open(os.path.join(folder_path,label_path,out_name[:-4] + '.txt'), 'w') as f:
            for label in out_labels:
                for item in label:
                    f.write(str(item))
                    f.write(' ')
                f.write('\n') 

    print(file,'done')
        
             
