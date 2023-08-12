import cv2
import argparse

import numpy as np 

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required=True, help = 'path to input image')
args = parser.parse_args()

image = cv2.imread(args.image)
image = (image/image.max() * 255).astype('uint8')

# saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
# (success, saliency_map) = saliency.computeSaliency(image)
# print(success)
# saliency_map = (saliency_map * 255).astype('uint8')
# cv2.imshow('image',image)
# cv2.imshow('output',saliency_map)
# cv2.waitKey(0)

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliency_map) = saliency.computeSaliency(image)

print(np.percentile(saliency_map,75))
thresh_map = (saliency_map > np.percentile(saliency_map,90)).astype('uint8')*255
# thresh_map = cv2.threshold(saliency_map.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

thresh_map = cv2.dilate(thresh_map,np.ones((10,10)))
thresh_map = cv2.erode(thresh_map,np.ones((9,9)))

cv2.imshow('image',image)
cv2.imshow('output',saliency_map)
cv2.imshow('thresh',thresh_map)
cv2.waitKey(0)

min_w = 50
min_h = 50
max_w = 200
max_h = 200
rois = 0
max_iters = 1000
max_rois = 0
min_area = 900

erode = thresh_map
# for i in range(min_iters):
#     rois = 0
#     (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(erode, 4, cv2.CV_32S)
#     for i in range(1, num_labels):
#         w = stats[i, cv2.CC_STAT_WIDTH]
#         h = stats[i, cv2.CC_STAT_HEIGHT]
#         area = stats[i, cv2.CC_STAT_AREA]
#         keep_width = w >= min_w and w <= max_w
#         keep_height = h >= min_h and w <= max_h
#         keep_area = area > min_area
#         if all((keep_width, keep_height, keep_area)):
#             rois +=1
#     if rois > max_rois:
#         max_rois = rois
#         out = erode.copy()
#     erode = cv2.erode(erode, np.array([[0,1,0],[1,1,1],[0,1,0]]).astype('uint8'))
all_rects = []
for i in range(max_iters):
    if erode.sum() == 0:
        print(i)
        break
    rois = 0
    cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        cv2.drawContours(erode,[cnt],-1,[255,255,255],-1)
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        keep_width = w >= min_w and w <= max_w
        keep_height = h >= min_h and h <= max_h
        keep_area = area > min_area
        if all((keep_width, keep_height, keep_area)):
            rois +=1
            all_rects.append([x,y,w,h])
    if rois > max_rois:
        max_rois = rois
        out = erode.copy()
    erode = cv2.erode(erode, np.array([[0,1,0],[1,1,1],[0,1,0]]).astype('uint8'))

roi_rects = []
# if out.any:
#     (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(out, 4, cv2.CV_32S)
#     tile_out = tile.copy()
#     for i in range(1, num_labels):
#         x = stats[i, cv2.CC_STAT_LEFT]
#         y = stats[i, cv2.CC_STAT_TOP]
#         w = stats[i,cv2.CC_STAT_WIDTH]
#         h = stats[i,cv2.CC_STAT_HEIGHT]
#         area = stats[i,cv2.CC_STAT_AREA]
#         center_x, center_y = centroids[i]
#         keep_width = w >= min_w and w <= max_w
#         keep_height = h >= min_h and h <= max_h
#         keep_area = area > min_area
#         if all((keep_width, keep_height,keep_area)):
#             roi_rects.append([x,y,w,h])
#             cv2.rectangle(tile_out, (x,y),(x+w,y+h),(0,255,0),3)
tile = image
tile_out = tile.copy()
try:
    cnts = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        keep_width = w >= min_w and w <= max_w
        keep_height = h >= min_h and h <= max_h
        keep_area = area > min_area
        if all((keep_width, keep_height,keep_area)):
            roi_rects.append([x,y,w,h])
            cv2.rectangle(tile_out, (x,y),(x+w,y+h),(0,255,0),3)    
except:
    out=False

# pltTiles(tile, thresh, tile_out)
   
plt.imshow(tile_out)
plt.show()



# saliency = cv2.saliency.ObjectnessBING_create()
# saliency.setTrainingPath('bing')
# (success, saliency_map) = saliency.computeSaliency(image)
# num_classes = saliency_map.shape[0]
# print(num_classes)
# print(saliency_map)

# min_w = 50
# min_h = 50
# max_w = 200
# max_h = 200
# rois = 0
# max_iters = 1000
# max_rois = 0
# min_area = 900
# saliency_map2 = []
# for rect in saliency_map:
#     x,y,x2,y2 = rect.flatten()
#     area = (x2-x)*(y2-y)
#     w = x2-x
#     h = y2-y
#     keep_width = w >= min_w and w <= max_w
#     keep_height = h >= min_h and h <= max_h
#     keep_area = area > min_area
#     if all((keep_width,keep_height,keep_area)):
#         saliency_map2.append(rect)
        

# output = image.copy()
# for i in range(0, min(num_classes,50)):
#     (start_x, start_y, end_x, end_y) = saliency_map2[i].flatten()
#     # output = image.copy()
#     color = np.random.randint(0,255,size=(3,))
#     color = [int(c) for c in color]
#     cv2.rectangle(output, (start_x, start_y), (end_x, end_y), color, 2)
#     cv2.imshow('image',output)
#     cv2.waitKey(0)
# plt.imshow(output)
# plt.show()