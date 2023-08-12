# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:20:45 2023

@author: kmccl
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

lonlats = np.load('17Rcz00321.npy')
lonlats[lonlats==0] = np.nan
plt.imshow(lonlats[:,:,0])

labels = np.load('17R_outboxes.npy')
im = cv2.imread('17Rcz00321.png')



for label in labels:
    left, top, right, bottom = label
    
    
    
    minlat = np.nanmin(lonlats[:,:,1])
    minlon = np.nanmin(lonlats[:,:,0])
    maxlat = np.nanmax(lonlats[:,:,1])
    maxlon = np.nanmax(lonlats[:,:,0])
    
    left_in = minlon < left < maxlon
    right_in = minlon < right < maxlon
    top_in = minlat < top < maxlat
    bottom_in = minlat < bottom < maxlat
    
    tl_in = left_in and top_in
    tr_in = right_in and top_in
    br_in = bottom_in and right_in
    bl_in = bottom_in and left_in
    
    if tl_in and br_in and bl_in and tr_in:
        lonlats = cv2.resize(lonlats,(2592,1944),interpolation=cv2.INTER_CUBIC)
    #box_in = tl_in and tr_in and br_in and bl_in
    #print(box_in)
    
    tl = (left, top)
    tr = (right, top)
    br = (right, bottom)
    bl = (left, bottom)
    
    tl_min_dist_idx = np.nanargmin(np.abs(lonlats-tl).sum(axis=2))
    tl_min_dist_idx = np.unravel_index(tl_min_dist_idx, lonlats.shape[:2])
    tr_min_dist_idx = np.nanargmin(np.abs(lonlats-tr).sum(axis=2))
    tr_min_dist_idx = np.unravel_index(tr_min_dist_idx, lonlats.shape[:2])
    bl_min_dist_idx = np.nanargmin(np.abs(lonlats-bl).sum(axis=2))
    bl_min_dist_idx = np.unravel_index(bl_min_dist_idx, lonlats.shape[:2])
    br_min_dist_idx = np.nanargmin(np.abs(lonlats-br).sum(axis=2))
    br_min_dist_idx = np.unravel_index(br_min_dist_idx, lonlats.shape[:2])
    br_min_dist = np.linalg.norm(br - lonlats[br_min_dist_idx[:2]])
    tr_min_dist = np.linalg.norm(tr - lonlats[tr_min_dist_idx[:2]])
    bl_min_dist = np.linalg.norm(bl - lonlats[bl_min_dist_idx[:2]])
    tl_min_dist = np.linalg.norm(tl - lonlats[tl_min_dist_idx[:2]])
    tl_in = tl_min_dist < 0.01
    tr_in = tr_min_dist < 0.01
    br_in = br_min_dist < 0.01
    bl_in = bl_min_dist < 0.01
    if tl_in and br_in and bl_in and tr_in:
        tl = np.min((tl_min_dist_idx, tr_min_dist_idx,
                     br_min_dist_idx, bl_min_dist_idx),axis=0)
        br = np.max((tl_min_dist_idx, tr_min_dist_idx,
                     br_min_dist_idx, bl_min_dist_idx),axis=0).T
        cv2.rectangle(im, (tl[1],tl[0]), (br[1],br[0]), [255,255,255],2)
plt.imshow(im)
