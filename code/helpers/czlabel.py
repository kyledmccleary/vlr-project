# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 00:43:19 2023

@author: kmccl
"""
import os
import cv2
import numpy as np
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


PATH = './cesims2'
BOXPATH = '_outboxes.npy'


# all_boxes = np.load(BOXPATH)

# label_path = 'datasets/11R/train/labels'
# out_path = 'datasets/11R/train/images'
# thumb_path = 'datasets/11R/train/thumbs'

train_test = 'test'

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', default = None)#required=True)
args = parser.parse_args()

# PATH = os.path.join(PATH,args.folder)
# files = os.listdir(PATH)

nets = ['10S','10T','11R','12R','16T','17R','17T','18S',
        '32S','32T','33S','33T','52S','53S','54S','54T']

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    idx = np.unravel_index(idx, array.shape)
    return idx

def get_box(lonlat_array, landmark_lonlats):
    left, top, right, bottom = landmark_lonlats
    
def dowork(folder):
    if folder not in nets:
        return
    path = os.path.join(PATH,folder)
    boxpath = folder + BOXPATH
    boxes = np.load(boxpath)
    files = os.listdir(path)
    for file in tqdm(files):
        if file.endswith('npy'):
            lonlats = np.load(os.path.join(path,file))
            lonlats[lonlats==0] = np.nan
            
            labels = []
            for i in range(len(boxes)):
                left, top, right, bottom = boxes[i]
                
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
                
                if (left_in or right_in) and (top_in or bottom_in):
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
                
                cnt = np.array([tl_min_dist_idx[::-1], tr_min_dist_idx[::-1],
                       br_min_dist_idx[::-1], bl_min_dist_idx[::-1]])
                
                x,y,w,h = cv2.boundingRect(cnt)
                if w > 50 and h > 50:
                    xc = x + w//2
                    xc_norm = xc / 2592
                    yc = y + h//2
                    yc_norm = yc / 1944
                    w_norm = w / 2592
                    h_norm = h / 1944
                    out_box = [i, xc_norm, yc_norm, w_norm, h_norm]
                    labels.append(out_box)
                
                br_min_dist = np.linalg.norm(br - lonlats[br_min_dist_idx[:2]])
                tr_min_dist = np.linalg.norm(tr - lonlats[tr_min_dist_idx[:2]])
                bl_min_dist = np.linalg.norm(bl - lonlats[bl_min_dist_idx[:2]])
                tl_min_dist = np.linalg.norm(tl - lonlats[tl_min_dist_idx[:2]])
                # tl_in = tl_min_dist < 0.01
                # if tl_in:
                #     tl_px = tl_min_dist_idx
                # tr_in = tr_min_dist < 0.01
                # if tr_in:
                #     tr_px = tr_min_dist_idx
                # br_in = br_min_dist < 0.01
                # if br_in:
                #     br_px = br_min_dist_idx
                # bl_in = bl_min_dist < 0.01
                # if bl_in:
                #     bl_px = bl_min_dist_idx
                # if tl_in and br_in and bl_in and tr_in:
                #     tl = np.min((tl_min_dist_idx, tr_min_dist_idx,
                #                  br_min_dist_idx, bl_min_dist_idx),axis=0)
                #     br = np.max((tl_min_dist_idx, tr_min_dist_idx,
                #                  br_min_dist_idx, bl_min_dist_idx),axis=0)
                #    # cv2.rectangle(im, (tl[1],tl[0]), (br[1],br[0]), [255,255,255],2)
                    

                #     top_px = tl[0]
                #     left_px = tl[1]
                #     bottom_px = br[0]
                #     right_px = br[1]
                    
                #     w = right_px - left_px
                #     h = bottom_px - top_px
                #     xc = (right_px + left_px)//2
                #     yc = (top_px + bottom_px)//2
                #     xc_norm = xc / 2592
                #     yc_norm = yc / 1944
                #     w_norm = w/2592
                #     h_norm = h/1944
                #     out_box = [i, xc_norm, yc_norm, w_norm, h_norm]
                #     labels.append(out_box)
                
                
                
                # y,x = find_nearest(box[:,:,0],left)
                # left_in = np.abs(box[y,x,0] - left) < 0.00005
                # y,x = find_nearest(box[:,:,0],right)
                # right_in = np.abs(box[y,x,0] - right) < 0.00005
                # y,x = find_nearest(box[:,:,1],top)
                # top_in = np.abs(box[y,x,1] - top) < 0.00005
                # y,x = find_nearest(box[:,:,1],bottom)
                # bottom_in = np.abs(box[y,x,1] - bottom) < 0.00005
                # if (left_in and right_in) and (top_in and bottom_in):
                #     big_box = cv2.resize(box, (2592, 1944), interpolation=cv2.INTER_CUBIC)
                #     left_idx = find_nearest(big_box[:,:,0], left)
                #     right_idx = find_nearest(big_box[:,:,0], right)
                #     bottom_idx = find_nearest(big_box[:,:,1], bottom)
                #     top_idx = find_nearest(big_box[:,:,1], top)
    
                #     idxs = np.vstack((left_idx, right_idx, bottom_idx, top_idx))
                #     mins = np.min(idxs,axis=0)
                #     maxes = np.max(idxs,axis=0)
                #     yc = (maxes[0]+mins[0])/2
                #     h = maxes[0]-mins[0]
                #     w = maxes[1]-mins[1]
                #     xc = (maxes[1]+mins[1])/2
                #     xc_norm = xc/2592
                #     yc_norm = yc/1944
                #     w_norm = w/2592
                #     h_norm = h/1944
                #     out_box = [i, xc_norm, yc_norm, w_norm, h_norm]
                #     labels.append(out_box)
                       
            name = file[:-4]
            im_name = name + '.png'
            im = cv2.imread(os.path.join(path,im_name))
            label_name = name + '.txt'
            
            base = 'datasets/' + folder + '/' + train_test
            if not os.path.exists(base):
                os.mkdir(base)
            
            label_path = base + '/labels'
            out_path = base + '/images'
            thumb_path = base + '/thumbs'
            
            if not os.path.exists(os.path.join(out_path)):
                os.mkdir(os.path.join(out_path))
                os.mkdir(os.path.join(thumb_path))
                os.mkdir(os.path.join(label_path))
                        
            cv2.imwrite(os.path.join(out_path,im_name),im)
            thumb = im.copy()
            thumb = cv2.resize(thumb, (320,240))
            for label in labels:
                l,x,y,w,h = label
                w = int(round(w*320))
                h = int(round(h*240))
                x = int(round(x*320) - w/2)
                y = int(round(y*240) - h/2)
                thumb = cv2.rectangle(thumb, (x,y), (x+w,y+h), [255, 255, 255], thickness=1)
            cv2.imwrite(os.path.join(thumb_path,im_name),thumb)
            with open(os.path.join(label_path,label_name),'w') as outfile:
                for label in labels:
                    for item in label:
                        outfile.write(str(item))
                        outfile.write(' ')
                    outfile.write('\n')
            print(name[:-4],'saved')
        
def main():
    if args.folder is not None:
        folders = [args.folder]
    else:
        folders = os.listdir(PATH)
       # folders.remove('99Z')
        # folders.remove('EW')
    
    p = Pool(cpu_count())
    p.map(dowork, folders)
    p.close()
    p.join()

if __name__ == '__main__':
    main()    
            
        