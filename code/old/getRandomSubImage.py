import rasterio
import numpy as np
import pyproj
from getMGRS import getMGRS
import cv2

# CRS = 'EPSG:3857'

def getRandomSubImage(im, lonlat_arr, key, width_m, height_m, gsd, segment = False):
    grid = getMGRS()
    grid_left, grid_bottom, grid_right, grid_top = grid[key]
    
    lat_range = grid_top - grid_bottom
    lon_range = grid_right - grid_left
    
    
    
    im_width = im.shape[1]
    im_height = im.shape[0]
    
    lon_arr = lonlat_arr[:,:,0]
    lat_arr = lonlat_arr[:,:,1]
    
    min_lon = lon_arr.min()
    max_lon = lon_arr.max()
    min_lat = lat_arr.min()
    max_lat = lat_arr.max()
    
    lon_diff = grid_left - min_lon
    lat_diff = grid_bottom - min_lat
    point_x = np.random.uniform(lon_diff,lon_range+lon_diff)
    point_y = np.random.uniform(lat_diff,lat_range+lat_diff)
    
    im_lon_range = max_lon - min_lon
    im_lat_range = max_lat - min_lat
    
    pt_x_px = int((point_x/im_lon_range)*im_width)
    pt_y_px = int((point_y/im_lat_range)*im_height)
    
    tl_x_px = int(pt_x_px - 0.5*width_m/gsd)
    tl_y_px = int(im_height - pt_y_px - 0.5*height_m/gsd)
    br_x_px = int(pt_x_px + 0.5*width_m/gsd)
    br_y_px = int(im_height - pt_y_px + 0.5*height_m/gsd)
    box = (tl_x_px, tl_y_px, br_x_px, br_y_px)
    
    rotated_im, rotated_lons, rotated_lats, rotation = randomRotation(im, lon_arr, lat_arr)
    cropped_im, cropped_lons, cropped_lats = boxCrop(rotated_im, rotated_lons, rotated_lats, box)
    
    min_x_loc_y, min_x_loc_x = np.where(cropped_lons == cropped_lons.min())
    max_x_loc_y, max_x_loc_x = np.where(cropped_lons == cropped_lons.max())
    min_y_loc_y, min_y_loc_x = np.where(cropped_lats == cropped_lats.min())
    max_y_loc_y, max_y_loc_x = np.where(cropped_lats == cropped_lats.max())
    
    min_lon_corner = cropped_lons[min_x_loc_y,min_x_loc_x],cropped_lats[min_x_loc_y,min_x_loc_x]
    max_lon_corner = cropped_lons[max_x_loc_y,max_x_loc_x],cropped_lats[max_x_loc_y,max_x_loc_x]
    min_lat_corner = cropped_lons[min_y_loc_y,min_y_loc_x],cropped_lats[min_y_loc_y,min_y_loc_x]
    max_lat_corner = cropped_lons[max_y_loc_y,max_y_loc_x],cropped_lats[max_y_loc_y,max_y_loc_x]
    
    min_lon_corner_lon_offset = min_lon_corner[0] - min_lon
    min_lon_corner_lat_offset = min_lon_corner[1] - min_lat
    max_lon_corner_lon_offset = max_lon_corner[0] - min_lon
    max_lon_corner_lat_offset = max_lon_corner[1] - min_lat
    min_lat_corner_lon_offset = min_lat_corner[0] - min_lon
    min_lat_corner_lat_offset = min_lat_corner[1] - min_lat
    max_lat_corner_lon_offset = max_lat_corner[0] - min_lon
    max_lat_corner_lat_offset = max_lat_corner[1] - min_lat
    
    left_x = min_lon_corner_lon_offset/im_lon_range
    left_y = 1 - min_lon_corner_lat_offset/im_lat_range
    right_x = max_lon_corner_lon_offset/im_lon_range
    right_y = 1 - max_lon_corner_lat_offset/im_lat_range
    bottom_x = min_lat_corner_lon_offset/im_lon_range
    bottom_y = 1 - min_lat_corner_lat_offset/im_lat_range
    top_x = max_lat_corner_lon_offset/im_lon_range
    top_y = 1 - max_lat_corner_lat_offset/im_lat_range
    
    label = (left_x,left_y,right_x,right_y,bottom_x,bottom_y,top_x,top_y)
    label = [l[0] for l in label]
                     
    
    if segment:
        zeros = np.zeros_like(lon_arr, dtype = np.uint8)
        zeros[tl_y_px:br_y_px,tl_x_px:br_x_px] = 1
        center_h = im_height//2
        center_w = im_width//2
        M = cv2.getRotationMatrix2D((center_w, center_h), -rotation, 1)
        mask_rot = cv2.warpAffine(zeros, M, (im_width, im_height))
        label = mask_rot
        
        
    
    return cropped_im, label


# def getRandomSubImage(raster, key, width_m, height_m):
#     grid = getMGRS()
#     left, bottom, right, top = grid[key]
    
#     ###########
#     crs = raster.crs
#     # crs = CRS
#     ###########
    
#     point_x = np.random.uniform(left,right)
#     point_y = np.random.uniform(bottom,top)
    
#     transformer1 = pyproj.Transformer.from_crs('EPSG:4326',crs,always_xy=True)
#     pt_m = transformer1.transform(point_x,point_y)
#     tl_x = pt_m[0] - width_m/2
#     tl_y = pt_m[1] + height_m/2
#     br_x = pt_m[0] + width_m/2
#     br_y = pt_m[1] - height_m/2
        
#     im_width = raster.width
#     im_height = raster.height 
    
#     im_left, im_bottom, im_right, im_top = raster.bounds
    
#     im_gsd = (im_right - im_left)/im_width
    
#     tl_x_px = int(round((tl_x - im_left)/im_gsd,-1))
#     tl_y_px = int(round(im_height - (tl_y - im_bottom)/im_gsd,-1))
#     br_x_px = int(round((br_x - im_left)/im_gsd,-1))
#     br_y_px = int(round(im_height - (br_y - im_bottom)/im_gsd,-1))
#     box = (tl_x_px, tl_y_px, br_x_px, br_y_px)    
    
#     cols,rows = np.meshgrid(np.arange(im_width),np.arange(im_height))
#     xs, ys = rasterio.transform.xy(raster.transform,rows,cols)
#     xs = np.array(xs)
#     ys = np.array(ys)
    
#     rgb = raster.read()
#     rgbT = np.transpose(rgb, (1,2,0))
    
#     rotated_im, rotated_xs, rotated_ys = randomRotation(rgbT,xs,ys)
#     cropped_im, cropped_xs, cropped_ys = boxCrop(rotated_im, rotated_xs, rotated_ys,box)
    
#     transformer2 = pyproj.Transformer.from_crs(crs, 'EPSG:4326', always_xy=True)
    
#     im_lon_min, im_lat_max = transformer2.transform(im_left,im_top)
#     im_lon_max, im_lat_min = transformer2.transform(im_right,im_bottom)
       
#     im_lon_min = left
#     im_lon_max = right
#     im_lat_min = bottom
#     im_lat_max = top
    
#     lon_range = im_lon_max - im_lon_min
#     lat_range = im_lat_max - im_lat_min
    
#     min_x_loc_y, min_x_loc_x = np.where(cropped_xs == cropped_xs.min())
#     max_x_loc_y, max_x_loc_x = np.where(cropped_xs == cropped_xs.max())
#     min_y_loc_y, min_y_loc_x = np.where(cropped_ys == cropped_ys.min())
#     max_y_loc_y, max_y_loc_x = np.where(cropped_ys == cropped_ys.max())
    
#     min_lon_corner = transformer2.transform(cropped_xs[min_x_loc_y,min_x_loc_x],cropped_ys[min_x_loc_y,min_x_loc_x])
#     max_lon_corner = transformer2.transform(cropped_xs[max_x_loc_y,max_x_loc_x],cropped_ys[max_x_loc_y,max_x_loc_x])
#     min_lat_corner = transformer2.transform(cropped_xs[min_y_loc_y,min_y_loc_x],cropped_ys[min_y_loc_y,min_y_loc_x])
#     max_lat_corner = transformer2.transform(cropped_xs[max_y_loc_y,max_y_loc_x],cropped_ys[max_y_loc_y,max_y_loc_x])
    
#     min_lon_corner_lon_offset = min_lon_corner[0] - im_lon_min
#     min_lon_corner_lat_offset = min_lon_corner[1] - im_lat_min
#     max_lon_corner_lon_offset = max_lon_corner[0] - im_lon_min
#     max_lon_corner_lat_offset = max_lon_corner[1] - im_lat_min
#     min_lat_corner_lon_offset = min_lat_corner[0] - im_lon_min
#     min_lat_corner_lat_offset = min_lat_corner[1] - im_lat_min
#     max_lat_corner_lon_offset = max_lat_corner[0] - im_lon_min
#     max_lat_corner_lat_offset = max_lat_corner[1] - im_lat_min
#     left_x = min_lon_corner_lon_offset/lon_range
#     left_y = min_lon_corner_lat_offset/lat_range
#     right_x = max_lon_corner_lon_offset/lon_range
#     right_y = max_lon_corner_lat_offset/lat_range
#     bottom_x = min_lat_corner_lon_offset/lon_range
#     bottom_y = min_lat_corner_lat_offset/lat_range
#     top_x = max_lat_corner_lon_offset/lon_range
#     top_y = max_lat_corner_lat_offset/lat_range
    
#     label = (left_x,left_y,right_x,right_y,bottom_x,bottom_y,top_x,top_y)
#     label = [l[0] for l in label]
    
#     return cropped_im, label
        
def randomRotation(image, xs, ys):
    image_h = image.shape[0]
    image_w = image.shape[1]
    center_h = image_h//2
    center_w = image_w//2
    rotation = np.random.uniform(-180,180)
    M = cv2.getRotationMatrix2D((center_w, center_h), rotation, 1)
    rotated_image = cv2.warpAffine(image, M, (image_w, image_h))
    rotated_xs = cv2.warpAffine(xs,M,(image_w,image_h))
    rotated_ys = cv2.warpAffine(ys,M,(image_w,image_h))
    return rotated_image, rotated_xs, rotated_ys, rotation

def boxCrop(image, xs, ys, box):
    tl_x_px, tl_y_px, br_x_px, br_y_px = box
    cropped_image = image[tl_y_px:br_y_px,tl_x_px:br_x_px]
    cropped_xs = xs[tl_y_px:br_y_px,tl_x_px:br_x_px]
    cropped_ys = ys[tl_y_px:br_y_px,tl_x_px:br_x_px]
    return cropped_image, cropped_xs, cropped_ys
    
    