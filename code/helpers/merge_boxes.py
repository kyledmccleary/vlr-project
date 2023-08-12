# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 00:21:48 2023

@author: kmccl
"""
import numpy as np

keys = ['10S', '10T', '11R', '11S', '12R', '16T', '17R', '17T', '18S', '32S', '32T', 
        '33S', '33T', '52S', '53S', '54S', '54T']

all_boxes = np.empty((0,4))
for key in keys:
    boxes = np.load(key + '_outboxes.npy')
    all_boxes = np.vstack((all_boxes, boxes))
    
np.save('all_boxes.npy',all_boxes)
