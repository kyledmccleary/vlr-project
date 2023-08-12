import cv2
import numpy as np
from multiprocessing import Pool,cpu_count
import matplotlib.pyplot as plt

im = cv2.imread('italy_test.png')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
gray = cv2.copyMakeBorder(gray, 495, 494, 664,663,cv2.BORDER_CONSTANT)
gray = cv2.resize(gray,(2000,2000))
gray = cv2.Canny(gray,0,255,cv2.THRESH_OTSU)


def getDistinctness(tile):
    res = cv2.matchTemplate(gray,tile,cv2.TM_CCOEFF_NORMED)
    max_val = res.max()
    max_idx = np.unravel_index(res.argmax(),res.shape)
    res[max_idx] = 0
    distinctness = max_val - res.max()
    return distinctness

def getTiles(size):
    cpy = gray.copy()
    slices = cpy.reshape(gray.shape[0]//sz,
                      sz,
                      gray.shape[1]//sz,
                      sz)
    slices = slices.swapaxes(1,2)
    slices_flat = slices.reshape((slices.shape[0]*slices.shape[1],sz,sz))
    slices_list = list(slices_flat)
    return slices_list




if __name__ == '__main__':
    
    windowsizes = [50, 100, 125, 250, 400, 500]
    out_arr = np.zeros(gray.shape)
    
    p = Pool(cpu_count())
    for sz in windowsizes:
        sz_out_arr = np.zeros(gray.shape)
        tiles = getTiles(sz)
        distinctnesses = p.starmap(getDistinctness,zip(tiles))
        distinctnesses = np.array(distinctnesses)
        distinctness_map = distinctnesses.reshape(gray.shape[0]//sz, gray.shape[1]//sz)
        for i in range(distinctness_map.shape[0]):
            for j in range(distinctness_map.shape[1]):
                sz_out_arr[i*sz:(i+1)*sz,j*sz:(j+1)*sz] += distinctness_map[i,j]
        sz_out_arr_thresh = sz_out_arr > np.percentile(sz_out_arr,90)
        out_arr += sz_out_arr_thresh*1
        print(sz, 'done')
    p.close()
    p.join()

    plt.imshow(out_arr)
    out_arr = (out_arr/out_arr.max() * 255).astype('uint8')
    out_im = cv2.addWeighted(gray,0.5,out_arr,0.5,1)
    plt.imshow(out_im)
