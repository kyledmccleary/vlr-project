import rasterio
import numpy as np
import pyproj

import cv2

import matplotlib.pyplot as plt

import os 

LAT_STEP = 1
LON_STEP = 1

def rotateImage(image,rotation):
    image_h = image.shape[0]
    image_w = image.shape[1]
    center_h = image_h//2
    center_w = image_w//2
    M = cv2.getRotationMatrix2D((center_w,center_h),rotation,1)
    rotatedImage = cv2.warpAffine(image, M,(image_w,image_h))
    return rotatedImage

def processLandsatImage(im_rgb):
    im_gray = cv2.cvtColor(im_rgb,cv2.COLOR_RGB2GRAY)
    thresh = im_gray > 0
    coords = np.column_stack(np.where(thresh))
    angle = cv2.minAreaRect(coords)[-1] * -1
    rotated = rotateImage(im_gray,angle)
    rotated_rgb = rotateImage(im_rgb,angle)
    r_thresh = np.uint8(rotated > 0)
    contours, hierarchy = cv2.findContours(r_thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contour = sorted(contours, key=cv2.contourArea)[-1]
    x,y,w,h = cv2.boundingRect(contour)
    out = rotated_rgb[y:y+h,x:x+w]   
    return out

key = '17T'

path = os.path.join('../data2',key)
path = os.path.join('../data2')

data_path = os.path.join(path,'data')
label_path = os.path.join(path,'labels')

for file in os.listdir(path):
    if file[-4:] == '.tif':       
        with rasterio.open(os.path.join(path,file)) as src:
            width = src.width
            height = src.height
            left, bottom, right, top = src.bounds
            CRS = src.crs
            transformer = pyproj.Transformer.from_crs(CRS,'EPSG:4326')
            lower_left = transformer.transform(left,bottom)
            lower_right = transformer.transform(right,bottom)
            upper_right = transformer.transform(right,top)
            upper_left = transformer.transform(left,top)
            rgb = src.read([1,2,3])
            rgbT = np.transpose(rgb,(1,2,0))
            rgb_out = processLandsatImage(rgbT)
            grid_left,grid_bottom,grid_right,grid_top = (-180,-90,180,90)
            lons = np.arange(grid_left,grid_right,LON_STEP)
            lats = np.arange(grid_bottom,grid_top,LAT_STEP)
            im_width = len(lons)
            im_height = len(lats)
            im_width_lon = grid_right - grid_left
            im_height_lat = grid_top - grid_bottom
            lower_left_px = [np.int32((lower_left[1]-grid_left)*im_width/im_width_lon), im_height - np.int32((grid_top-lower_left[0])*im_height/im_height_lat)]
            lower_right_px = [np.int32((lower_right[1]-grid_left)*im_width/im_width_lon),im_height -np.int32( (grid_top-lower_right[0])*im_height/im_height_lat)]
            upper_left_px = [np.int32((upper_left[1]-grid_left)*im_width/im_width_lon), im_height -np.int32((grid_top-upper_left[0])*im_height/im_height_lat)]
            upper_right_px = [np.int32((upper_right[1]-grid_left)*im_width/im_width_lon), im_height -np.int32((grid_top-upper_right[0])*im_height/im_height_lat)]
            
            pts = np.int32([lower_left_px,lower_right_px,upper_right_px,upper_left_px])
            
            label = np.zeros((im_height,im_width),dtype = np.uint8)
            label = cv2.polylines(label,[pts],True,255,1)
            
            cv2.imwrite(os.path.join(label_path,file[:-4]+'_label.png'),label)
            bgr_out = cv2.cvtColor(rgb_out*255,cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(data_path,file[:-4]+'_data.png'),bgr_out)
            plt.imshow(label)            
            print(file,'done')
    