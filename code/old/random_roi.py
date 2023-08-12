import cv2
import csv
import matplotlib.pyplot as plt

name = 'florida_random_boxes'
mask = cv2.imread('florida_mask.jpg',cv2.IMREAD_GRAYSCALE)
mask_w = mask.shape[1]
mask_h = mask.shape[0]

tile_h = 200
tile_w = 200

ver_tiles = mask_h//tile_h
hor_tiles = mask_w//tile_w

if mask_w - (hor_tiles*tile_w) > tile_w/2:
    hor_tiles += 1
if mask_h - (ver_tiles*tile_h) > tile_h/2:
    ver_tiles += 1

im_min_lon=-88
im_min_lat =24
im_max_lon =-77
im_max_lat =31

lon_range = im_max_lon - im_min_lon
lat_range = im_max_lat - im_min_lat
lon_per_px = lon_range/mask_w
lat_per_px = lat_range/mask_h


out_boxes = []
out_mask = mask.copy()
for i in range(ver_tiles):
    for j in range(hor_tiles):
        tl_x = j*200
        tl_y = i*200
        br_x = j*200 + 200
        br_y = i*200 + 200
        print(mask[tl_y:br_y,tl_x:br_x].sum())

        if mask[tl_y:br_y,tl_x:br_x].sum() > 0:
            
            cv2.rectangle(out_mask,(tl_x,tl_y),(br_x,br_y),[100,100,100],3) 
            tl_lon = lon_per_px*tl_x + im_min_lon
            tl_lat = im_max_lat - lat_per_px*tl_y 
            br_lon = lon_per_px*br_x + im_min_lon
            br_lat = im_max_lat - lat_per_px*br_y
            out_boxes.append([tl_lon,tl_lat,br_lon,br_lat])

with open(name + '.csv','w') as csvfile:
    writer = csv.writer(csvfile,delimiter=',')
    header = ['min_lon','min_lat','max_lon','max_lat']
    writer.writerow(header)
    writer.writerows(out_boxes)
    
plt.imshow(out_mask)
