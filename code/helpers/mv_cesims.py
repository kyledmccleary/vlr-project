# -*- coding: utf-8 -*-
"""
Created on Sat Jul 29 14:14:50 2023

@author: kmccl
"""

import os
import argparse
from tqdm import tqdm

BASEPATH = './cesims2'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=True, type=str)
    args = parser.parse_args()
    return args
        

def main():
    args = parse_args()
    files = os.listdir(os.path.join(BASEPATH,args.dir))
    for file in tqdm(files):
        if file.endswith('npy'):
            key = file[:3]
            pic = file[:-3] + 'png'
            if not os.path.exists(os.path.join(BASEPATH,key)):
                os.mkdir(os.path.join(BASEPATH,key))
                print('made dir',os.path.join(BASEPATH,key))
            if os.path.exists(os.path.join(BASEPATH,key,file)):
                newfile = file[:-4] + '-' + args.dir + file[-4:]
                newpic = pic[:-4] + '-' + args.dir + pic[-4:]
            else:
                newfile = file
                newpic = pic
            os.rename(os.path.join(BASEPATH,args.dir,file), os.path.join(BASEPATH,key,newfile))
            os.rename(os.path.join(BASEPATH,args.dir,pic), os.path.join(BASEPATH,key,newpic))


if __name__ == '__main__':
    main()