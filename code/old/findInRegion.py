import cv2
import rasterio
import rasterio.plot
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


path = 'italy_2023-01-01-2023-03-30_lt50cl.tif'
with rasterio.open(path) as src:
        #rasterio.plot.show(src)
        width = src.width
        height = src.height
        r = np.uint8(src.read(1))
        g = np.uint8(src.read(2))
        b = np.uint8(src.read(3))
        cols,rows = np.meshgrid(np.arange(width),np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows,cols)
        lons = np.array(xs)
        lats = np.array(ys)
im_rgb = np.stack((r,g,b),axis=-1)
im_gray = cv2.cvtColor(im_rgb,cv2.COLOR_RGB2GRAY)

sift = cv2.SIFT_create()
kp = sift.detect(im_gray,None)
#im_rgb = cv2.drawKeypoints(im_gray,kp,im_rgb,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('1',im_rgb)

test_template = cv2.imread('Screenshot 2023-03-03 120330.png')
test_template_gray = cv2.cvtColor(test_template,cv2.COLOR_BGR2GRAY)
kp,desc = sift.detectAndCompute(test_template_gray,None)
#test_template = cv2.drawKeypoints(test_template_gray,kp,test_template,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow('2',test_template)

def gsdToScale(template_min_gsd,template_max_gsd,image_gsd = 2000):
    min_scale = template_min_gsd/image_gsd
    max_scale = template_max_gsd/image_gsd
    return min_scale, max_scale

def scaleImage(image,scale):
    image_h = image.shape[0]
    image_w = image.shape[1]
    scaled_h = image_h * scale
    scaled_w = image_w * scale
    scaledImage = cv2.resize(image,(round(scaled_w),round(scaled_h)))
    return scaledImage

def rotateImage(image,rotation):
    image_h = image.shape[0]
    image_w = image.shape[1]
    center_h = image_h//2
    center_w = image_w//2
    M = cv2.getRotationMatrix2D((center_w,center_h),rotation,1)
    rotatedImage = cv2.warpAffine(image, M,(image_w,image_h))
    return rotatedImage

def templateMatch(index,iterable):
    return
 
   

METHOD = cv2.TM_CCOEFF_NORMED
min_gsd = 300
max_gsd = 500
image_gsd = 1000
min_scale, max_scale = gsdToScale(min_gsd,max_gsd,image_gsd=image_gsd)
overall_max = 0
for scale in [0.4,0.45]:
    for rotation in [-2,-1.5,-1,0,1,1.5,2,3]:
        rotated_im = deepcopy(im_gray)
        rotated_im = rotateImage(rotated_im,rotation)
        scaled_template = deepcopy(test_template_gray)
        scaled_template = scaleImage(test_template_gray,scale)
        res = cv2.matchTemplate(rotated_im,scaled_template,method=METHOD)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > overall_max:
            overall_max = max_val
            overall_max_loc = max_loc
            best_scale = scale
            best_rot = rotation
            print(overall_max)
max_loc = overall_max_loc
scaled_template_rgb = scaleImage(test_template,best_scale)
rotated_image = rotateImage(im_rgb,best_rot)
rotated_image[max_loc[1]:max_loc[1]+scaled_template_rgb.shape[0],max_loc[0]:max_loc[0]+scaled_template_rgb.shape[1]] = scaled_template_rgb
im_bgr = cv2.cvtColor(rotated_image,cv2.COLOR_RGB2BGR)
cv2.imshow('',im_bgr)