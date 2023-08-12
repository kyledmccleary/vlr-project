import numpy as np
import os
from getMGRS import getMGRS
from PIL import Image
from tqdm import tqdm
import rasterio
import pyproj
import cv2

grid = getMGRS()
regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T',
           '18S', '32S', '32T', '33S', '33T', '52S', '53S',
           '54S', '54T']

base_folder = 'F:\landsat_images'
folders = os.listdir(base_folder)
folders = [folder for folder in folders 
           if os.path.isdir(os.path.join(base_folder, folder))]
out_path = './multi_label/train'
out_crs = 'EPSG:4326'
# folders.remove('elsweyr')

for folder in tqdm(folders):
    files = os.listdir(os.path.join(base_folder, folder))
    for file in tqdm(files):
        with rasterio.open(os.path.join(base_folder,folder,file)) as src:
            crs = src.crs
            im = np.uint8(src.read().transpose((1,2,0)))
            im_width = src.width
            im_height = src.height
            left, bottom, right, top = src.bounds
        if folder == 'elsweyr':
            label = np.zeros(len(regions), dtype=np.uint8)
            imfile = file[:-4] + '.png'
            labfile = file[:-4] + '.npy'
            im = Image.fromarray(im)   
            min_dim = min(im_width, im_height)
            scale = min_dim/360
            new_w = int(round(im_width/scale))
            new_h = int(round(im_height/scale))
        
            small_im = im.resize((new_w,new_h))
            small_im.save(os.path.join(out_path,'images',imfile))
            np.save(os.path.join(out_path,'labels',labfile), label)
        
        else:
            transformer = pyproj.Transformer.from_crs(crs, out_crs, always_xy=True)
            left_lon, top_lat = transformer.transform(left, top)
            right_lon, bot_lat = transformer.transform(right, bottom)
            tl = (left_lon, top_lat)
            tr = (right_lon, top_lat)
            br = (right_lon, bot_lat)
            bl = (left_lon, bot_lat)
            label = np.zeros(len(regions), dtype=np.uint8)
            for i in range(len(label)):
                key = regions[i]
                left, bottom, right, top = grid[key]
                tl_in = left < left_lon < right and bottom < top_lat < top
                tr_in = left < right_lon < right and bottom < top_lat < top
                br_in = left < right_lon < right and bottom < bot_lat < top
                bl_in = left < left_lon < right and bottom < bot_lat < top
                reg_in = tl_in or tr_in or br_in or bl_in
                if reg_in:
                    label[i] = 1
            imfile = file[:-4] + '.png'
            labfile = file[:-4] + '.npy'
            im = Image.fromarray(im)   
            min_dim = min(im_width, im_height)
            scale = min_dim/360
            new_w = int(round(im_width/scale))
            new_h = int(round(im_height/scale))
            
            small_im = im.resize((new_w,new_h))
            small_im.save(os.path.join(out_path,'images',imfile))
            np.save(os.path.join(out_path,'labels',labfile), label)