import cv2
import argparse

import numpy as np

import matplotlib.pyplot as plt

from getMGRS import getMGRS

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', required = False, help = 'path to input image')
parser.add_argument('-t', '--thresh', default = 90, type=float, help = 'threshold percentile')
parser.add_argument('-o', '--outpath', default = 'boxes.csv', type=str, help = 'path to output file')
parser.add_argument('-b', '--bounds', default = [-88, 24, -77, 31], nargs='+', type=float, help = 'min lon, min lat, max lon, max lat')
parser.add_argument('-w', '--window', default = 50, type=int, help = 'pixel width of box around each POI')
parser.add_argument('-s', '--save', default = False, type=bool, help = 'save saliency map')
parser.add_argument('-n', '--name', default = 'unnmamed', type = str)
parser.add_argument('-k', '--key', default = None)
args = parser.parse_args()

# im = cv2.imread(args.image)



if args.key:
    grid = getMGRS()
    args.bounds = grid[args.key]






whole = np.zeros((21600*2,21600*4,3),dtype='uint8')

a1 = cv2.imread('../world.200412.3x21600x21600.A1.jpg')
whole[0:21600,0:21600] = a1
del(a1)

a2 = cv2.imread('../world.200412.3x21600x21600.A2.jpg')
whole[21600:,0:21600] = a2
del(a2)

b1 = cv2.imread('../world.200412.3x21600x21600.B1.jpg')
whole[0:21600,21600:21600*2] = b1
del(b1)

b2 = cv2.imread('../world.200412.3x21600x21600.B2.jpg')
whole[21600:,21600:21600*2] = b2
del(b2)

c1 = cv2.imread('../world.200412.3x21600x21600.C1.jpg')
whole[0:21600,21600*2:21600*3] = c1
del(c1)

c2 = cv2.imread('../world.200412.3x21600x21600.C2.jpg')
whole[21600:,21600*2:21600*3] = c2
del(c2)

d1 = cv2.imread('../world.200412.3x21600x21600.D1.jpg')
whole[0:21600, 21600*3:21600*4] = d1
del(d1)

d2 = cv2.imread('../world.200412.3x21600x21600.D2.jpg')
whole[21600:, 21600*3:21600*4] = d2
del(d2)

bm = whole
del(whole)
bm_w = bm.shape[1]
bm_h = bm.shape[0]
# bm_w = whole.shape[1]
# bm_h = bm.shape[0]


left, bottom, right, top = args.bounds
im_lon_w = right - left
im_lon_h = top - bottom
bm_min_lon = -180
bm_min_lat = -90
bm_lon_w = 360
bm_lat_w = 180

start_x = round(((left - bm_min_lon)/bm_lon_w)*bm_w)
start_y = round((1-(top - bm_min_lat)/bm_lat_w)*bm_h)
end_x = round(((right - bm_min_lon)/bm_lon_w)*bm_w)
end_y = round((1-(bottom - bm_min_lat)/bm_lat_w)*bm_h)

im = bm[start_y:end_y, start_x:end_x]

im = cv2.resize(im,(im.shape[1]//2,im.shape[0]//2))

im_height = im.shape[0]
im_width = im.shape[1]

print('here')
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliency_map) = saliency.computeSaliency(im)
if args.save:
    cv2.imwrite(args.name + '_saliencymap.jpg',saliency_map*255)
    np.save(args.name + '_saliencymap.npy',saliency_map)
# perc_val = np.percentile(saliency_map, args.thresh)
# thresh_map = (saliency_map > perc_val).astype('uint8')*255
# cv2.imshow('im',im)
# cv2.imshow('output', saliency_map)
# cv2.imshow('thresh', thresh_map)
# cv2.waitKey(0)

# window = args.window
# threshcopy = thresh_map.copy()
# all_rects = []
# for i in range(im_height):
#     for j in range(im_width):
#         if thresh_map[i,j]:
#             cv2.rectangle(threshcopy,(j - window//2, i - window//2),(j+window//2,i+window//2),(100,100,100),3)
#             # cv2.circle(threshcopy,(j,i),window,(100,100,100),3)
#             all_rects.append((j - window//2,i-window//2,window,window))

# cv2.imshow('threshboxes',threshcopy)
# cv2.waitKey(0)

# # tuplify
# def tup(point):
#     return (point[0], point[1]);

# # returns true if the two boxes overlap
# def overlap(source, target):
#     # # unpack points
#     tl1, br1 = source;
#     tl2, br2 = target;

#     # checks
#     if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
#         return False;
#     if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
#         return False;
#     return True;
#     # x_a = max(tl1[0],tl2[0])
#     # y_a = max(tl1[1],tl2[1])
#     # x_b = min(br1[0],br2[0])
#     # y_b = min(br1[1],br2[1])
#     # inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
#     # src_area = (br1[0]-tl1[0] + 1) * (br1[1]-tl1[1] + 1)
#     # tgt_area = (br2[0]-tl2[0] + 1) * (br2[1] - tl2[1] + 1)
#     # iou = inter_area / float(src_area + tgt_area - inter_area)
#     # if iou > 0.5:
#     #     return True
#     # else:
#     #     return False

# # returns all overlapping boxes
# def getAllOverlaps(boxes, bounds, index):
#     overlaps = [];
#     for a in range(len(boxes)):
#         if a != index:
#             if overlap(bounds, boxes[a]):
#                 overlaps.append(a);
#     return overlaps;



# # go through the contours and save the box edges
# boxes = []
# for rect in all_rects:
#     x,y,w,h = rect
#     boxes.append([[x,y],[x+w,y+h]])

# # filter out excessively large boxes
# filtered = [];
# max_area = 300000;
# for box in boxes:
#     w = box[1][0] - box[0][0];
#     h = box[1][1] - box[0][1];
#     if w*h < max_area:
#         filtered.append(box);
# boxes = filtered;

# # go through the boxes and start merging
# merge_margin = 0

# # this is gonna take a long time
# finished = False
# highlight = [[0,0], [1,1]]
# points = [[[0,0]]]
# while not finished:
#     # set end con
#     finished = True

#     # check progress
#     # print("Len Boxes: " + str(len(boxes)));

#     # # draw boxes # comment this section out to run faster
#     # copy = np.copy(im);
#     # for box in boxes:
#     #     cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0,200,0), 1);
#     # cv2.rectangle(copy, tup(highlight[0]), tup(highlight[1]), (0,0,255), 2);
#     # for point in points:
#     #     point = point[0];
#     #     cv2.circle(copy, tup(point), 4, (255,0,0), -1);
#     # cv2.imshow("Copy", copy);
#     # key = cv2.waitKey(1);
#     # if key == ord('q'):
#     #     break;

#     # loop through boxes
#     index = len(boxes) - 1
#     while index >= 0:
#         # grab current box
#         curr = boxes[index]

#         # add margin
#         tl = curr[0][:]
#         br = curr[1][:]
        
#         merge_margin_x = -(br[0]-tl[0])//5
#         merge_margin_y = -(br[1]-tl[1])//5
        
#         tl[0] -= merge_margin_x
#         tl[1] -= merge_margin_y
#         br[0] += merge_margin_x
#         br[1] += merge_margin_y

#         # get matching boxes
#         overlaps = getAllOverlaps(boxes, [tl, br], index)
        
#         # check if empty
#         if len(overlaps) > 0:
#             # combine boxes
#             # convert to a contour
#             con = [];
#             overlaps.append(index);
#             for ind in overlaps:
#                 tl, br = boxes[ind];
#                 con.append([tl]);
#                 con.append([br]);
#             con = np.array(con);

#             # get bounding rect
#             x,y,w,h = cv2.boundingRect(con);

#             # stop growing
#             w -= 1;
#             h -= 1;
#             merged = [[x,y], [x+w, y+h]];

#             # highlights
#             highlight = merged[:];
#             points = con;

#             # remove boxes from list
#             overlaps.sort(reverse = True);
#             for ind in overlaps:
#                 del boxes[ind];
#             boxes.append(merged);

#             # set flag
#             finished = False;
#             break;

#         # increment
#         index -= 1;
# cv2.destroyAllWindows();

# # show final
# copy = np.copy(im);
# for box in boxes:
#     cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0,200,0), 3);
# cv2.imshow("Final", copy);
# cv2.waitKey(0);

# plt.imshow(copy)
# plt.show()