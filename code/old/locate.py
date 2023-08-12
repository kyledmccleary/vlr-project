import cv2
import rasterio
import os
import numpy as np
from copy import deepcopy
from pyproj import CRS
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge

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

roi_ims_path = 'roi_ims'
tif_path = '../tifs'

roi_ims = {}

KEY = '12R'

roi_im_files = os.listdir(roi_ims_path)
#roi_im_files = os.listdir('../test_saral')
#roi_ims_path = '../test_saral'

for roi_im_file in roi_im_files:
    if roi_im_file[-4:] == '.tif' and roi_im_file[:len(KEY)] == KEY:
        with rasterio.open(os.path.join(roi_ims_path, roi_im_file)) as src:
            with rasterio.open(os.path.join(roi_ims_path, '11R_spr.tif')) as src2:
                mosaic,transform = merge((src,src2))
                
                mosaic = mosaic.transpose(1,2,0)
                
                merge_meta = src.meta.copy()
                merge_meta.update({"driver":'GTiff',
                    "height":mosaic.shape[1],
                    "width":mosaic.shape[2],
                    "transform":transform,
                    })
                
                
                # cv2.imshow('test',mosaic)
                r = np.uint8(src.read(1))
                g = np.uint8(src.read(2))
                b = np.uint8(src.read(3))
                bgr = np.stack((b,g,r),axis=-1)  
                # bgr = cv2.copyMakeBorder(bgr,100,100,100,100,cv2.BORDER_CONSTANT)
                key = roi_im_file[:-4]
                
                #######
                bgr = mosaic
                ########
                
                
                roi_ims[key] = bgr
                gsd = abs(rasterio.transform.xy(src.transform, 1, 1)[1] - rasterio.transform.xy(src.transform, 0, 0)[1])

#################TESTING#########
# im_path = '../tifs/11Rtif/11R_1.tiff'  
im_path = '../test_saral/11R_3.tiff'
    
#################################

with rasterio.open(im_path) as src:
   # rasterio.plot.show(src.read([1,2,3]))
    r = np.uint8(src.read(1))
    g = np.uint8(src.read(2))
    b = np.uint8(src.read(3))
    bgr = np.stack((b,g,r),axis=-1)
    im = bgr
    
#########
cam_gsd = 157
image_gsd = 1000
scale = cam_gsd / image_gsd
tmp = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

omax = 0
rotations = np.linspace(-180, 180, 3)
scales = np.linspace(0.157,0.3,2)

for roi_im in roi_ims:
    for scale in scales:
        for rotation in rotations:
            test_im = cv2.cvtColor(roi_ims[roi_im], cv2.COLOR_BGR2GRAY)
            test_im = deepcopy(test_im)
            test_im = rotateImage(test_im, rotation)
            test_tmp = deepcopy(tmp)
            test_tmp = scaleImage(tmp,scale)
            res = cv2.matchTemplate(test_im,test_tmp, method = cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > omax:
                omax = max_val
                oloc = max_loc
                okey = roi_im
                best_im = deepcopy(test_im)
                best_tmp = deepcopy(test_tmp)
out = best_im
new_height = best_tmp.shape[0]
new_width = best_tmp.shape[1]
out[oloc[1]:oloc[1]+new_height,oloc[0]:oloc[0]+new_width] = best_tmp
out = cv2.rectangle(out,(oloc[0],oloc[1]),(oloc[0]+new_width,oloc[1]+new_height),color=[150,150,255])
cv2.imshow(okey, out)


with rasterio.open(os.path.join(roi_ims_path,okey+'.tif')) as src:
    width = src.width
    height = src.height
    lons = np.zeros((height,width))
    lats = np.zeros((height,width))
    cols,rows = np.meshgrid(np.arange(width),np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows,cols)
    x3857 = np.array(xs)
    y3857 = np.array(ys)
    transformer = Transformer.from_crs(3857, 4326)
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
            int(src.height*scale),
            int(src.width*scale)
        ),
        resampling = Resampling.bilinear
    )
    
    transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )    
    width = int(src.width*scale)
    height = int(src.height*scale)
    cols,rows = np.meshgrid(np.arange(width),np.arange(height))
    xs, ys = rasterio.transform.xy(transform, rows,cols)
    x3857 = np.array(xs)
    y3857 = np.array(ys)
    lons_tmp = np.zeros((height,width))
    lats_tmp = np.zeros((height,width))
    
    transformer = Transformer.from_crs(3857, 4326)
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
