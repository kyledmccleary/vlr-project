# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 02:18:45 2023

@author: kmccl
"""

import os
import argparse
import numpy as np
from getMGRS import getMGRS
from tqdm import tqdm

BASEPATH = '.'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=True, type=str)
    args = parser.parse_args()
    return args

def sort_array(lonlat):
    grid = getMGRS()
    lonlat_nan = lonlat
    lonlat_nan[lonlat_nan==0] = np.nan
    avlon = np.nanmean(lonlat_nan[:, :, 0])
    avlat = np.nanmean(lonlat_nan[:, :, 1])
    lonlat_key = '99Z'
    for key in grid:
        left, bottom, right, top = grid[key]
        if left < avlon < right and bottom < avlat < top:
            lonlat_key = key
    return lonlat_key     
        

def main():
    args = parse_args()
    files = os.listdir(os.path.join(BASEPATH,args.dir))
    cnt = 0
    for file in tqdm(files):
        if file.endswith('npy'):
            if file.startswith('5000'):
                key = '99Z'
            else:
                key = sort_array(np.load(os.path.join(BASEPATH,args.dir,file)))
            pic = file[:-3] + 'png'
            new_name = key + 'cz' + str(cnt).zfill(5)
            new_pic = new_name + '.png'
            new_file = new_name + '.npy'
            os.rename(os.path.join(BASEPATH,args.dir,file), os.path.join(BASEPATH,args.dir,new_file))
            os.rename(os.path.join(BASEPATH,args.dir,pic), os.path.join(BASEPATH,args.dir,new_pic))
            cnt += 1

if __name__ == '__main__':
    main()