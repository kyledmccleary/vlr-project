# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 03:55:27 2023

@author: kmccl
"""

import numpy as np
from getMGRS import getMGRS
import cv2

grid = getMGRS()
im = cv2.imread('world.200412.3x5400x2700.jpg')
w = im.shape[1]
h = im.shape[0]

outim = im.copy()
outim = np.zeros((2700,5400,4),dtype='uint8')
for key in grid:
    left, top, right, bottom = grid[key]
    left = left + 180
    right = right + 180
    top = 90 - top
    bottom = 90 - bottom
    left = left/360
    right = right/360
    top = top/180
    bottom = bottom/180
    left = int(left*w)
    right = int(right*w)
    top = int(top*h)
    bottom = int(bottom*h)
    cv2.rectangle(outim,(left,top),(right,bottom),(0,255,0,255),3)
cv2.imshow('',outim)
cv2.imwrite('mgrsfig.png',outim)