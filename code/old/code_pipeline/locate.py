import cv2
import rasterio
import os
import numpy as np
from copy import deepcopy
from pyproj import CRS
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
import rasterio.fill
from rasterio.features import sieve
import rasterio.plot
import opts

ROI_IMS_PATH = opts.ROI_IMS_PATH

def scaleImage(image,scale):
    image_h, image_w = image.shape
    scaled_h = image_h * scale
    scaled_w = image_w * scale
    scaledImage = cv2.resize(image,(int(scaled_w),int(scaled_h)))
    return scaledImage

def rotateImage(image,rotation):
    image_h, image_w = image.shape
    center_h = image_h//2
    center_w = image_w//2
    M = cv2.getRotationMatrix2D((center_w,center_h),rotation,1)
    rotatedImage = cv2.warpAffine(image, M,(image_w,image_h))
    return rotatedImage

def locateInTile(im_path, pred):
    roi_ims = {}
    roi_im_files = os.listdir(ROI_IMS_PATH)    
    for roi_im_file in roi_im_files:
        if roi_im_file[-4:] == '.tif' and roi_im_file[:len(pred)] == pred:
            with rasterio.open(os.path.join(ROI_IMS_PATH, roi_im_file),'r+') as src:                    
                    arr_im = src.read()
                    rgb_im = np.transpose(arr_im,(1,2,0))
                    rgb_im_int = np.uint8(rgb_im)
                    if rgb_im_int.max() <= 1:
                        rgb_im = rgb_im*255
                    rgb_im = np.uint8(rgb_im)
                    
                    key = roi_im_file[:-4]                   
                    roi_ims[key] = rgb_im
                    gsd = abs(rasterio.transform.xy(src.transform, 1, 1)[1] - rasterio.transform.xy(src.transform, 0, 0)[1])
    with rasterio.open(im_path) as src:
        arr_tmp = src.read()
        rgb_tmp = np.transpose(arr_tmp,(1,2,0))
        rgb_tmp_int = np.uint8(rgb_tmp)
        if rgb_tmp_int.max() <= 1:
            rgb_tmp = rgb_tmp*255
        rgb_tmp = np.uint8(rgb_tmp)
    
    cam_gsd = opts.CAM_GSD
    image_gsd = opts.ROI_IMAGE_GSD
    scale = cam_gsd / image_gsd
    tmp = cv2.cvtColor(rgb_tmp, cv2.COLOR_RGB2GRAY)
    
    omax = 0
    # rotations = np.linspace(-180, 180, 73)
    ####TESTING#####
    rotations = np.linspace(-180,180,3)
    scales = np.linspace(scale-scale*.1,scale+scale*.1,5)
    for roi_im in roi_ims:
        for scale in scales:
            for rotation in rotations:
                test_im = cv2.cvtColor(roi_ims[roi_im], cv2.COLOR_RGB2GRAY)
                test_im = deepcopy(test_im)
                # test_im = rotateImage(test_im, rotation)
                test_tmp = deepcopy(tmp)
                test_tmp = scaleImage(tmp,scale)
                test_tmp = rotateImage(test_tmp,scale)
                if(test_tmp.shape[1] <= test_im.shape[1] and test_tmp.shape[0] <= test_im.shape[0]):
                    res = cv2.matchTemplate(test_im,test_tmp, method = cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    if max_val > omax:
                        omax = max_val
                        oloc = max_loc
                        okey = roi_im
                        best_im = deepcopy(test_im)
                        best_tmp = deepcopy(test_tmp)
                        best_scale = scale
                        best_rotation = rotation
    new_height = best_tmp.shape[0]
    new_width = best_tmp.shape[1]
    out = best_im
    out[oloc[1]:oloc[1]+new_height,oloc[0]:oloc[0]+new_width] = best_tmp
    out = cv2.rectangle(out,(oloc[0],oloc[1]),(oloc[0]+new_width,oloc[1]+new_height),color=[150,150,255])
    
    
    with rasterio.open(os.path.join(ROI_IMS_PATH,okey+'.tif')) as src:
        width = src.width
        height = src.height
        lons = np.zeros((height,width))
        lats = np.zeros((height,width))
        cols,rows = np.meshgrid(np.arange(width),np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows,cols)
        x3857 = np.array(xs)
        y3857 = np.array(ys)
        CRS = src.crs
        transformer = Transformer.from_crs(CRS, 4326)
        for i in range(width):
            for j in range(height):
                lat,lon = transformer.transform(x3857[j,i],y3857[j,i])
                lons[j,i] = lon
                lats[j,i] = lat
    lons_im_tmp = lons[oloc[1]:oloc[1]+new_height,oloc[0]:oloc[0]+new_width]
    lats_im_tmp = lats[oloc[1]:oloc[1]+new_height,oloc[0]:oloc[0]+new_width]
        
    with rasterio.open(im_path) as src:
        data = src.read(
            out_shape=(
                src.count,
                int(src.height*best_scale),
                int(src.width*best_scale)
            ),
            resampling = Resampling.bilinear
        )
        
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )    
        width = int(src.width*best_scale)
        height = int(src.height*best_scale)
        cols,rows = np.meshgrid(np.arange(width),np.arange(height))
        xs, ys = rasterio.transform.xy(transform, rows,cols)
        x3857 = np.array(xs)
        y3857 = np.array(ys)
        CRS = src.crs
        lons_tmp = np.zeros((height,width))
        lats_tmp = np.zeros((height,width))
        
        transformer = Transformer.from_crs(CRS, 4326)
        for i in range(width):
            for j in range(height):
                lat,lon = transformer.transform(x3857[j,i],y3857[j,i])
                lons_tmp[j,i] = lon
                lats_tmp[j,i] = lat
    
    sqdiffs_lons = (lons_im_tmp - lons_tmp)**2 
    lons_sum_sqdiff = sqdiffs_lons.sum()
    sqdiffs_lats = (lats_im_tmp - lats_tmp)**2
    lats_sum_sqdiff = sqdiffs_lats.sum()
    print(lons_sum_sqdiff, lats_sum_sqdiff)
    avgdiff_lons = abs(lons_im_tmp - lons_tmp).sum()/(width*height)
    avgdiff_lats = abs(lats_im_tmp - lats_tmp).sum()/(width*height)
    print('Lon error: ',avgdiff_lons)
    print('Lat error: ', avgdiff_lats)

    return avgdiff_lons, avgdiff_lats,omax
