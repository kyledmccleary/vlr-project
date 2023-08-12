###multiclass label

import numpy as np
import os
from getMGRS import getMGRS
from tqdm import tqdm
from PIL import Image

grid = getMGRS()
regions = ['10S', '10T', '11R', '12R', '16T', '17R', '17T',
           '18S', '32S', '32T', '33S', '33T', '52S', '53S',
           '54S', '54T']

base_folder = './cesims2'
folders = os.listdir(base_folder)
out_path = './multi_label/test'

for folder in tqdm(folders):
    files = os.listdir(os.path.join(base_folder,folder))
    for file in tqdm(files):
        if file.endswith('.npy'):
            lonlats = np.load(os.path.join(base_folder,folder,file))
            lonlats[lonlats == 0] = np.nan
            label = np.zeros(len(regions), dtype=np.uint8)
            for i in range(len(label)):
                key = regions[i]
                left, bottom, right, top = grid[key]
                left_in = left < lonlats[:,:,0]
                right_in = right > lonlats[:,:,0]
                top_in = top > lonlats[:,:,1]
                bottom_in = bottom < lonlats[:,:,1]
                lon_in = left_in * right_in
                lat_in = top_in * bottom_in
                point_in = lon_in * lat_in
                if point_in.sum() > 0:
                    label[i] = 1
            imfile = file[:-4] + '.png'
            im = Image.open(os.path.join(base_folder, folder, 
                                         imfile))
            small_im = im.resize((480, 360))
            small_im.save(os.path.join(out_path,'images',imfile))
            np.save(os.path.join(out_path,file),'labels',label)
            