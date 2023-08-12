import cv2
import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.merge import merge

from numpy.lib.stride_tricks import sliding_window_view as swv

import argparse

from multiprocessing import Pool
from tqdm import tqdm



parser = argparse.ArgumentParser(description='whichimage')
parser.add_argument('--n', '--tifnum')
args = parser.parse_args()

tifnum = args.n
if tifnum:
    tifnum = int(tifnum)

src1 = rasterio.open('tif1.tiff')
src2 = rasterio.open('tif2.tiff')
src3 = rasterio.open('tif3.tiff')

im1 = src1.read()
im2 = src2.read()
im3 = src3.read()

merged_image, transform = merge((src1,src2,src3))
merged_image_t = merged_image.transpose((1,2,0))
merged_image_bgr = cv2.cvtColor(merged_image_t, cv2.COLOR_RGB2BGR)
src1.close()
src2.close()
src3.close()

im1_t = im1.transpose((1,2,0))
im2_t = im2.transpose((1,2,0))
im3_t = im3.transpose((1,2,0))
im1_bgr = cv2.cvtColor(im1_t, cv2.COLOR_RGB2BGR)
im2_bgr = cv2.cvtColor(im2_t, cv2.COLOR_RGB2BGR)
im3_bgr = cv2.cvtColor(im3_t, cv2.COLOR_RGB2BGR)


gray = cv2.cvtColor(merged_image_t, cv2.COLOR_RGB2GRAY)
gray1 = cv2.cvtColor(im1_bgr, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(im2_bgr, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(im3_bgr, cv2.COLOR_BGR2GRAY)

window_size = 250
tiles = swv(gray, (window_size,window_size))  
tiles1 = swv(gray1, (window_size,window_size))
tiles2 = swv(gray2, (window_size,window_size))
tiles3 = swv(gray3, (window_size,window_size))

# tiles_flat = tiles.reshape(tiles.shape[0]*tiles.shape[1], tiles.shape[2],tiles.shape[3]) 

distinctness_arr = np.zeros(gray.shape)
distinctness1 = np.zeros(gray1.shape)
distinctness2 = np.zeros(gray2.shape)
distinctness3 = np.zeros(gray3.shape)

if tifnum == 1:
    gray = gray1
elif tifnum == 2:
    gray = gray2
elif tifnum == 3:
    gray = gray3
else:
    gray = cv2.resize(gray,(gray.shape[1]//10,gray.shape[0]//10))
    window_size = 25
    tiles = swv(gray, (window_size,window_size))  
    print('no selection')

tuples = []
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        tuples.append((i,j))
np.save('tuples'+str(tifnum),tuples)
tot = len(tuples)

def getDistinctness(index,pt):
    row = pt[0]
    col = pt[1]
    res = cv2.matchTemplate(gray, tiles[row,col], cv2.TM_CCOEFF_NORMED)
    max_val = res.max()
    res[np.where(res==max_val)] = -100
    distinctness = max_val - res.max()
    print(index,'of',tot,'done')
    return distinctness

classic = False
if classic:
    darray = np.zeros(gray.shape)
    for i in tqdm(range(gray.shape[0]-25)):
        for j in range(gray.shape[1]-25):
            res = cv2.matchTemplate(gray, tiles[i,j], cv2.TM_CCOEFF_NORMED)
            max_val = res.max()
            res[np.where(res==max_val)] = -100
            darray[i+12,j+12] += max_val - res.max()




if __name__ == '__main__':
    p = Pool(3)
    distinctness_list = p.starmap(getDistinctness,enumerate(tuples))    
    p.join()
    p.close()
    np.save('dlist'+str(tifnum),distinctness_list)