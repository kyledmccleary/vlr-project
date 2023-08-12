# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 20:13:01 2023

@author: kmccl
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool, cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key')
parser.add_argument('-m', '--mode')
args = parser.parse_args()



BASE_PATH = '.\\datasets\\' + args.key + '\\' + args.mode
REGION = '.'
LABEL_PATH = 'labels'
IMAGE_PATH = 'images'

#section images and bounding boxes
def section_image_and_labels(filename):
    label_path = os.path.join(BASE_PATH, LABEL_PATH)
    image_path = os.path.join(BASE_PATH, IMAGE_PATH)
    region_im_path = os.path.join(image_path, REGION)
    region_lab_path = os.path.join(label_path, REGION)
    filepath = os.path.join(region_im_path,filename)
    image = cv2.imread(os.path.join(region_im_path, filename))
    if image.shape[0] != 1944 or image.shape[1] != 2592:
        return
    labels = []
    with open(os.path.join(region_lab_path, 
                           filename[:-4] + '.txt')) as infile:
        for line in infile:
            labels.append(line)
    
    label_arr = []
    for label in labels:
        label_strings = label.split()
        label_class = int(label_strings[0])
        label_xc = np.float64(label_strings[1])
        label_yc = np.float64(label_strings[2])
        label_w = np.float64(label_strings[3])
        label_h = np.float64(label_strings[4])
        label_arr.append([label_class, label_xc, label_yc, label_w, label_h])
    label_arr = np.array(label_arr)
    
    image_section1 = image[0:1296,0:1296].copy()
    image_section2 = image[0:1296,1296:2592].copy()
    image_section3 = image[648:1944,0:1296].copy()
    image_section4 = image[648:1944,1296:2592].copy()
    
    h, w, _ = image.shape
    labels_s1 = []
    labels_s2 = []
    labels_s3 = []
    labels_s4 = []
    for label in label_arr:
        s1 = get_section_label(label, 0, 1296, 0, 1296)
        if s1 is not None:
            labels_s1.append(s1)
        
        s2 = get_section_label(label, 0, 1296, 1296, 2592)
        if s2 is not None:
            labels_s2.append(s2)
        
        s3 = get_section_label(label, 648, 1944, 0, 1296)
        if s3 is not None:
            labels_s3.append(s3)
        
        s4 = get_section_label(label, 648, 1944, 1296, 2592)
        if s4 is not None:
            labels_s4.append(s4)    
    
    image_section1_filename = filename[:-4] + '_s1.png'
    label_section1_filename = filename[:-4] + '_s1.txt'
    image_section1_filepath = os.path.join(region_im_path,
                                           image_section1_filename)
    label_section1_filepath = os.path.join(region_lab_path,
                                           label_section1_filename)
    save_section(image_section1, labels_s1, image_section1_filepath,
                 label_section1_filepath)
    
    image_section2_filename = filename[:-4] + '_s2.png'
    label_section2_filename = filename[:-4] + '_s2.txt'
    image_section2_filepath = os.path.join(region_im_path,
                                           image_section2_filename)
    label_section2_filepath = os.path.join(region_lab_path,
                                           label_section2_filename)
    image_section3_filename = filename[:-4] + '_s3.png'
    label_section3_filename = filename[:-4] + '_s3.txt'
    image_section3_filepath = os.path.join(region_im_path,
                                           image_section3_filename)
    label_section3_filepath = os.path.join(region_lab_path,
                                           label_section3_filename)
    image_section4_filename = filename[:-4] + '_s4.png'
    label_section4_filename = filename[:-4] + '_s4.txt'
    image_section4_filepath = os.path.join(region_im_path,
                                           image_section4_filename)
    label_section4_filepath = os.path.join(region_lab_path,
                                           label_section4_filename)
    save_section(image_section2, labels_s2, image_section2_filepath,
                 label_section2_filepath)
    save_section(image_section3, labels_s3, image_section3_filepath,
                 label_section3_filepath)
    save_section(image_section4, labels_s4, image_section4_filepath,
                 label_section4_filepath)
    
    os.remove(filepath)
    labpath = os.path.join(region_lab_path, filename[:-4] + '.txt')
    os.remove(labpath)
    
    
def save_section(image, labels, im_filepath, lab_filepath):
    
    with open(lab_filepath, 'w') as outfile:
        for label in labels:
            for item in label:
                outfile.write(str(item))
                outfile.write(' ')
            outfile.write('\n')
    cv2.imwrite(im_filepath, image)
       
        
def get_section_label(label, min_y, max_y, min_x, max_x):
    min_y_px = min_y
    max_y_px = max_y
    min_x_px = min_x
    max_x_px = max_x
    
    cl, xc, yc, w, h = label
    xc_px = int(round(xc*2592))
    yc_px = int(round(yc*1944))
    w_px = int(round(w*2592))
    h_px = int(round(h*1944))
    
    top = yc_px - h_px//2
    bottom = top + h_px
    left = xc_px - w_px//2
    right = left + w_px
    
    if top < min_y_px:
        top = min_y_px
    elif top > max_y_px:
        top = max_y_px
    if left < min_x_px:
        left = min_x_px
    elif left > max_x_px:
        left = max_x_px
    if bottom < min_y_px:
        bottom = min_y_px
    elif bottom > max_y_px:
        bottom = max_y_px
    if right < min_x_px:
        right = min_x_px
    elif right > max_x_px:
        right = max_x_px
    
    if top != bottom and left != right:
        yc_s = (top + bottom) / 2 - min_y_px
        xc_s = (left + right) / 2 - min_x_px
        w_s = right - left
        h_s = bottom - top
        yc_s_out = yc_s/(max_y_px - min_y_px)
        h_s_out = h_s/(max_y_px - min_y_px)
        xc_s_out = xc_s/(max_x_px - min_x_px)
        w_s_out = w_s/(max_x_px - min_x_px)
        return([cl, xc_s_out, yc_s_out, w_s_out, h_s_out])
    else:
        return None
    
    
if __name__ == '__main__':
    filenames_all = os.listdir(os.path.join(BASE_PATH, IMAGE_PATH, REGION))
    filenames = []
    for filename in filenames_all:
        if not filename.startswith('l8'):
            filenames.append(filename)
    p = Pool(cpu_count())
    p.starmap(section_image_and_labels,zip(filenames))
    p.close()
    p.join()