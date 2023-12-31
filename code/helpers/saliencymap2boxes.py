import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import argparse
from getMGRS import getMGRS

grid = getMGRS()

parser = argparse.ArgumentParser()

parser.add_argument('-k', '--key', required=True)
parser.add_argument('-w', '--window', default = 100, type=int, help = 'pixel width of box around each POI')
# parser.add_argument('-s', '--save', default = True, type=bool, help = 'save saliency map')
parser.add_argument('-n', '--num_boxes', default = 20, type = int)
args = parser.parse_args()


saliency_map = np.load(args.key + '_saliencymap.npy')

im_h = saliency_map.shape[0]
im_w = saliency_map.shape[1]

window = args.window
num_boxes = args.num_boxes
boxes = []

for n in tqdm(range(num_boxes)):

    max_sum = 0
    for i in range(im_h-window):
        for j in range(im_w-window):
            if saliency_map[i:i+window,j:j+window].sum() > max_sum:
                max_sum = saliency_map[i:i+window,j:j+window].sum()
                tl_x = j
                tl_y = i
    boxes.append([tl_x, tl_y, window, window])
    # if n == 0:
    #     fig1 = saliency_map.copy()
    #     cv2.rectangle(fig1, (tl_x,tl_y),(tl_x+window,tl_y+window),(1,1,1),3)
    # if n == 1:
    #     fig2 = saliency_map.copy()
    #     cv2.rectangle(fig2, (tl_x,tl_y),(tl_x+window,tl_y+window),(1,1,1),3)
    # if n == 24:
    #     fig3 = saliency_map.copy()
    #     cv2.rectangle(fig3, (tl_x,tl_y),(tl_x+window,tl_y+window),(1,1,1),3)
    saliency_map[tl_y:tl_y+window,tl_x:tl_x+window] = 0
    
    
saliency_map = cv2.imread(args.key + '_saliencymap.jpg')
# # np.save('boxessm.npy',np.array(boxes))    
for box in boxes:
    x,y,w,h = box
    cv2.rectangle(saliency_map,(x,y),(x+w,y+h),(1,1,1),3)
cv2.imwrite(args.key + '_boxes.jpg', saliency_map)
# ax = plt.subplot(141)
# ax.imshow(fig1)
# ax.set_title('n=1',size=40)
# ax.set_xticks(ticks=[])
# ax.set_yticks(ticks=[])
# ax = plt.subplot(142)
# ax.imshow(fig2)
# ax.set_title('n=2',size=40)
# ax.set_xticks(ticks=[])
# ax.set_yticks(ticks=[])
# ax = plt.subplot(143)
# ax.imshow(fig3)
# ax.set_title('n=25',size=40)
# ax.set_xticks(ticks=[])
# ax.set_yticks(ticks=[])
# ax = plt.subplot(144)
# ax.imshow(saliency_map)
# ax.set_title('Output LOIs',size=40)
# ax.set_xticks(ticks=[])
# ax.set_yticks(ticks=[])
# plt.tight_layout()

# saliency_map = np.load('saliencymap.npy')    
# plt.imshow(saliency_map)
# plt.show()
# cv2.imshow('', saliency_map)
# cv2.waitKey(0)

left, bottom,right,top = grid[args.key]

outboxes = []
lon_range = right - left
lat_range = top - bottom
for box in boxes:
    tl_x, tl_y, box_w, box_h = box
    tl_lon = left + lon_range * (tl_x/im_w)
    tl_lat = bottom + lat_range * (1-tl_y/im_h)
    br_lon = left + lon_range * ((tl_x+window)/im_w)
    br_lat = bottom + lat_range * (1-(tl_y+window)/im_h)
    outboxes.append([tl_lon,tl_lat,br_lon,br_lat])

outboxes = np.array(outboxes)    
np.save(args.key + '_outboxes.npy',outboxes)