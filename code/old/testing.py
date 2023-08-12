import fiona
import rasterio
import cv2
import pyproj
import numpy as np
import os
import csv

SHAPE_PATH = 'shape/test.shp'

def rotateImage(image,rotation):
    image_h = image.shape[0]
    image_w = image.shape[1]
    center_h = image_h//2
    center_w = image_w//2
    M = cv2.getRotationMatrix2D((center_w,center_h),rotation,1)
    rotatedImage = cv2.warpAffine(image, M,(image_w,image_h))
    return rotatedImage, M

def rotateLandsat(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    thresh = gray>0
    coords = np.column_stack(np.where(thresh))
    angle = cv2.minAreaRect(coords)[-1] * -1
    rotated_rgb, M = rotateImage(im, angle)
    return rotated_rgb, M
    
def cropLandsat(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    r_thresh = np.uint8(gray>0)
    contours, hierarchy = cv2.findContours(r_thresh, mode=cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_NONE)
    contour = sorted(contours, key=cv2.contourArea)[-1]
    x,y,w,h = cv2.boundingRect(contour)
    out_im = im[y:y+h,x:x+w]
    return out_im, x,y,w,h

def getGeoms(path):
    with fiona.open(path) as shapefile:
        labels = [feature['properties']['id'] for feature in shapefile]
        geoms = [feature["geometry"] for feature in shapefile]
        shp_crs = shapefile.crs
    return geoms, labels, shp_crs

## Read GEOTiff and get relevant information
def readGEOTiff(path):
    with rasterio.open(path) as src:
        crs = src.crs
        im = src.read().transpose((1,2,0))
        im_width = src.width
        im_height = src.height
        im_width_crs = src.bounds[2] - src.bounds[0]
        im_height_crs = src.bounds[3] - src.bounds[1]
        left, bottom, right, top = src.bounds
    return im, crs, left, bottom, im_width, im_height, im_width_crs, im_height_crs
 
def rotateAndCrop(im):
    im = np.uint8(im*255)
    im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
    rotated, M = rotateLandsat(im)
    cropped, crop_x, crop_y, crop_w, crop_h = cropLandsat(rotated)
    return cropped, crop_x, crop_y, crop_w, crop_h, M

def transformToPixels(shp_crs, crs, geoms, left, bottom, im_width, im_height, im_width_crs, im_height_crs, M, crop_x, crop_y, cropped):
    transformer = pyproj.Transformer.from_crs(shp_crs,crs,always_xy = True) 
    rects = []
    polys = []
    for geom in geoms:
        coords = geom.coordinates[0]
        pixel_shape = []
        for coord in coords:
            x,y = transformer.transform(coord[0],coord[1])
            offset_x = x - left
            offset_y = y - bottom
            px_offset_x = round(offset_x/im_width_crs*im_width)
            px_offset_y = round((1-offset_y/im_height_crs)*im_height)
            pixel_shape.append((px_offset_x, px_offset_y))
        pixel_shape = np.array([[coord] for coord in pixel_shape])
        rot_pshape = cv2.transform(pixel_shape, M)
        pixel_shape = []
        for coord in rot_pshape:
            coord = coord[0]
            x, y = coord[0] - crop_x, coord[1] - crop_y
            pixel_shape.append((x,y))
        x, y, w, h = cv2.boundingRect(np.array(pixel_shape))
        rects.append([x,y,w,h])
        polys.append(pixel_shape)
    return rects, polys

def labelGEOTiff(folder,path, geoms, shp_crs):
    filepath = os.path.join(folder,path)
    im, crs, left, bottom, im_width, im_height, im_width_crs, im_height_crs = readGEOTiff(filepath)
    cropped, crop_x, crop_y, crop_w, crop_h, M = rotateAndCrop(im)
    cv2.imwrite(folder+'/images/'+path[:-4] + '.png', cropped)     
    
    rects, polys = transformToPixels(shp_crs, crs, geoms, left, bottom, im_width, im_height, im_width_crs, im_height_crs, M, crop_x, crop_y, cropped)
        
    labels = []
    out_polys = []
    for i in range(len(rects)): 
        x,y,w,h = rects[i]
        halfw = w//4
        halfh = h//4
        if (x > halfw and x < crop_w - halfw) or (x+w > halfw and x+w < crop_w - halfw):
            if (y>halfh and y<crop_h-halfh) or (y+h > halfh and y+h < crop_h-halfh):
                
                labels.append([i, x, y, w, h])
                out_polys.append([i] + polys[i])
                cropped = cv2.rectangle(cropped,(x,y),(x+w,y+h),[0,0,255],thickness=2)
    cropped = cv2.resize(cropped,(cropped.shape[1]//2,cropped.shape[0]//2))
    cv2.imwrite(folder+'/thumbs/' + path[:-4] + '.png',cropped)
    
    with open(folder+ '/labels/' + path[:-4] + '.txt', 'w') as f:
        for label in labels:
            for item in label:
                f.write(str(item))
                f.write(' ')
            f.write('\n')
    with open(folder+ '/polygons/' + path[:-4] + '.csv', 'w') as f:
        writer = csv.writer(f,delimiter=',')
        for out_poly in out_polys:
            writer.writerow(out_poly)
    print(path,'done')

def main():
    tif_folder = 'datasets/l8_fl'
    files = os.listdir(tif_folder)
    tif_files = []
    for file in files:
        if file.endswith('.tif'):
            tif_files.append(file)
    
    # tif_files = tif_files[:10]
    shape_path = SHAPE_PATH
    geoms, labels, shp_crs = getGeoms(shape_path)
    
    for tif_file in tif_files:
        labelGEOTiff(tif_folder, tif_file, geoms, shp_crs)  
       
if __name__ == '__main__':
    main()

