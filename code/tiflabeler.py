import rasterio
import cv2
import numpy as np
import os
import csv
import pyproj
from multiprocessing import Pool, cpu_count

# box_path = 'florida_algo_boxes.csv'
box_path = 'outboxes.npy'
box_crs = 'EPSG:4326'
infolder = 'NIR_test'
label_path = 'datasets/NIR/test/labels'
out_path = 'datasets/NIR/test/images'
thumb_path = 'datasets/NIR/test/thumbs'
batch_size = 1000
delete = False

if box_path.endswith('.csv'):
    boxes = []
    names = []
    with open(box_path,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter = ',')
        first = True
        
        ###
        idx = 0
        #####
        
        for row in reader:
            if row:
                if first:
                    header = row
                    first = False
                else:
                    box = row
                    # label = int(box[0])
                    # name = box[1]
                    # minlon = float(box[2])
                    # minlat = float(box[3])
                    # maxlon = float(box[4])
                    # maxlat = float(box[5])
                    
                    ########
                    minlon = float(row[0])
                    minlat = float(row[1])
                    maxlon = float(row[2])
                    maxlat = float(row[3])
                    label = idx
                    idx += 1
                    name = 'idk'
                    #####
                    
                    boxes.append(np.array([label, minlon,minlat, maxlon,maxlat]))
                    names.append(name)
    object_boxes = np.array(boxes)
elif box_path.endswith('.npy'):
    object_boxes = np.load(box_path)
    classes = len(object_boxes)
    arr = np.arange(0,classes)
    object_boxes = np.hstack((arr[:,np.newaxis], object_boxes))


def getTifs(infolder):
    files = os.listdir(infolder)
    tifs = []
    for file in files:
        if file.endswith('.tif'):
            tifs.append(file)
    return tifs

def readGEOTiff(path):
    with rasterio.open(os.path.join(infolder,path)) as src:
        crs = src.crs     
        if path.startswith('MODIS'):
            crs = 'EPSG:3857'
            im = np.uint8(src.read().transpose((1,2,0)))
        if path.startswith('l8'):
            # crs = 'EPSG:3857'
            im = np.uint8(src.read().transpose((1,2,0)))
        im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        im_width = src.width
        im_height = src.height
        im_width_crs = src.bounds[2] - src.bounds[0]
        im_height_crs = src.bounds[3] - src.bounds[1]
        left, bottom, right, top = src.bounds
    return im, crs, left, bottom, im_width, im_height, im_width_crs, im_height_crs

def transformToPixels(shp_crs, crs, boxes, left, bottom, im_width, im_height, im_width_crs, im_height_crs):
    transformer = pyproj.Transformer.from_crs(shp_crs,crs,always_xy = True) 
    px_boxes = []
    for box in boxes:
        label = box[0]
        tl_lon = box[1]
        tl_lat = max(box[4],box[2])
        br_lon = box[3]
        br_lat = min(box[4],box[2])
        tl_x, tl_y = transformer.transform(tl_lon, tl_lat)
        br_x, br_y = transformer.transform(br_lon, br_lat)        
        offset_tl_x = tl_x - left
        offset_tl_y = tl_y - bottom
        px_offset_tl_x = round(offset_tl_x/im_width_crs * im_width)
        px_offset_tl_y = round((1-offset_tl_y/im_height_crs)*im_height)       
        offset_br_x = br_x - left
        offset_br_y = br_y - bottom
        px_offset_br_x = round(offset_br_x/im_width_crs * im_width)
        px_offset_br_y = round((1-offset_br_y/im_height_crs)*im_height)        
        x = px_offset_tl_x
        y = px_offset_tl_y
        w = px_offset_br_x - x
        h = px_offset_br_y - y   
        px_boxes.append(np.array([label,x,y,w,h]))
    return np.array(px_boxes)
        
def getInbounds(im, im_width, im_height, boxes):
    labels = []
    for i in range(len(boxes)):
        label, x,y,w,h = boxes[i]
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        quarter_width = w//4
        quarter_height = h//4
        
        #check if at least one vertical side of the box is in the image
        if(x > quarter_width and x < im_width - quarter_width) or (x + w>quarter_width and x+w < im_width - quarter_width):
            #check if at least one horizontal side of the box is in the image
            if(y > quarter_height and y < im_height - quarter_height) or (y + h > quarter_height and y+h < im_height - quarter_height):
                #check that the pixels are not all nodata
                # if x < 0:
                #     w += x
                #     x= 0
                # if y < 0:
                #     h += y
                #     y= 0
                # if y+h > im_height:
                #     h -= (y+h - im_height)
                # if x+w > im_width:
                #     w -= (x +w - im_width)
                # if (im[x:x+w,y:y+h] == 0).sum() < 0.5*w*h*im.shape[2]:
                labels.append(boxes[i])
    return np.array(labels)
                
def getDataset(index,tif):
    im, crs, left, bottom, im_width, im_height, im_width_crs, im_height_crs = readGEOTiff(tif)
    px_boxes = transformToPixels(box_crs, crs, object_boxes, left, bottom, im_width, im_height, im_width_crs, im_height_crs)
    labels = getInbounds(im,im_width, im_height, px_boxes)
    print('tif',index,'done')
    return [im, labels, tif]

def normalizeLabels(im, labels):
    outlabels = []
    for label in labels:
         im_width = im.shape[1]
         im_height = im.shape[0]
         
         l = int(label[0])
         x = int(label[1])
         y = int(label[2])
         w = int(label[3])
         h = int(label[4])
         
         x_center = (x + x + w)/2.0
         y_center = (y + y + h)/2.0
         
         if y<0:
             h = y+h
             if h < 0:
                 h = 0
             y_center = h/2.0
             if y_center < 0:
                 y_center = 0
         if x < 0:
             w = x + w
             if w < 0:
                 w = 0
             x_center = w/2.0
             if x_center < 0:
                 x_center = 0
         if x + w > im_width:
             w = im_width - x
             if w<0:
                 w = 0
             x_center = (x + im_width)/2.0
             if x_center > im_width:
                 x_center = im_width
         if y+h > im_height:
            h = im_height - y
            if h<0:
                h=0
            y_center = (y+im_height)/2.0
            if y_center > im_height:
                y_center = im_height
         xc = x_center/im_width
         yc = y_center/im_height
         w = w/im_width
         h = h/im_height
         outlabel = [l, xc, yc, w, h]
         outlabels.append(outlabel)
       
    return outlabels
            
        
   
def normalizeAndSaveData(data):
    im = data[0]
    labels = data[1]
    tif = data[2]
    name = tif[:-4] + '.png'
    label_name = name[:-4] + '.txt'
    cv2.imwrite(os.path.join(out_path,name),im)
    thumb = im.copy()
    for label in labels:
        l,x,y,w,h = [int(val) for val in label]
        thumb = cv2.rectangle(thumb,(x,y),(x+w,y+h),[0,0,255],thickness=2)
    thumb = cv2.resize(thumb,(thumb.shape[1]//2,thumb.shape[0]//2))
    cv2.imwrite(os.path.join(thumb_path,name),thumb)
    outlabels = normalizeLabels(im,labels)
    with open(os.path.join(label_path,label_name),'w') as outfile:
        for outlabel in outlabels:
            for item in outlabel:
                outfile.write(str(item))
                outfile.write(' ')
            outfile.write('\n')
    print(name[:-4],'saved')
    if delete:
        os.remove(os.path.join(infolder,tif))
        


tifs = getTifs(infolder)

if __name__ == '__main__':
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(out_path, 'folder created')    
    if not os.path.exists(label_path):
        os.makedirs(label_path)
        print(label_path, 'folder created')
    if not os.path.exists(thumb_path):
        os.makedirs(thumb_path)
        print(thumb_path, 'folder created')
    
    num_tifs = len(tifs)
    batches = num_tifs//batch_size + 1
    
    p = Pool(cpu_count())
    for i in range(batches):
        if i == batches-1:
            tif_batch = tifs[batch_size*i:]
        else:
            tif_batch = tifs[batch_size*i:batch_size*(i+1)]
        dataset = p.starmap(getDataset,enumerate(tif_batch))
        
        dataset_iterable = zip(dataset)
        p.starmap(normalizeAndSaveData,dataset_iterable)
        # for data in dataset:
        #     normalizeAndSaveData(data)
    p.close()
    p.join()
      