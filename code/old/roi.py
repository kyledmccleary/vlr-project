import numpy as np
import cv2
import matplotlib.pyplot as plt
import csv



def getROIs(tile):
    tile_width = tile.shape[1]
    tile_height = tile.shape[0]
    
    tile_k = tile.reshape((-1,1)).astype('float32')
    K = 3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    km = cv2.kmeans(tile_k,K,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    ret,label,center = km
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((tile.shape))
    plt.imshow(res2)
    
    tile = res2
    
    pyramid_size = 10
    
    filter_size = 41
    contrast_filter = np.ones((filter_size,filter_size))*-1 / (filter_size*filter_size)
    contrast_filter[filter_size//2,filter_size//2] = 1 
    
    all_dog = np.zeros(tile.shape)
    rgbc = np.zeros((tile.shape[0],tile.shape[1]))
    
    r1 = tile[:,:,2].astype('float32')
    g1 = tile[:,:,1].astype('float32')
    b1 = tile[:,:,0].astype('float32')
    r1c = np.abs(cv2.filter2D(r1,-1,contrast_filter))
    g1c = np.abs(cv2.filter2D(g1,-1,contrast_filter))
    b1c = np.abs(cv2.filter2D(b1,-1,contrast_filter))
    rgbc += r1c + g1c + b1c
       
    
    for i in range(1,pyramid_size+1):
        smaller_im = cv2.resize(tile,(tile_width//(2*i),tile_height//(2*i)))
        smaller_im_bigger = cv2.resize(smaller_im,(tile_width,tile_height))
        bigger_im = cv2.resize(tile,(tile_width//(max(1,2*(i-1))),tile_height//(max(1,2*(i-1)))))
        bigger_im = cv2.resize(bigger_im,(tile_width,tile_height))
        r = smaller_im_bigger[:,:,2].astype('float32')
        g = smaller_im_bigger[:,:,1].astype('float32')
        b = smaller_im_bigger[:,:,0].astype('float32')
        rc = np.abs(cv2.filter2D(r,-1,contrast_filter))
        gc = np.abs(cv2.filter2D(g,-1,contrast_filter))
        bc = np.abs(cv2.filter2D(b,-1,contrast_filter))
        rgbc += rc+bc+gc
        dog = np.abs(bigger_im - smaller_im_bigger)
        all_dog += dog
        
    all_dog = all_dog.sum(axis=-1)
    all_dog = all_dog - all_dog.min()
    all_dog = all_dog/all_dog.max()
    rgbc = (rgbc-rgbc.min())/(rgbc-rgbc.min()).max()
    
    combo = all_dog*.3 + rgbc*.7
    combo = (combo/combo.max() * 255).astype('uint8')
    pltTiles(all_dog,rgbc,combo)
    perc = np.percentile(combo,90)
    val, thresh = cv2.threshold(combo, perc, 255,cv2.THRESH_OTSU & cv2.THRESH_BINARY)
    dil = cv2.dilate(thresh,np.ones((5,5),'uint8'))
    erode = cv2.erode(dil,np.ones((4,4),'uint8'))
    cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
            cv2.drawContours(erode,[cnt],-1,[255,255,255],-1)
    erode = cv2.GaussianBlur(erode,(31,31),0)
    val, erode = cv2.threshold(erode, 200,255, cv2.THRESH_BINARY)
    # dil = cv2.dilate(erode,np.ones((5,5),'uint8'))
    # erode = cv2.erode(erode,np.ones((4,4),'uint8'))
    pltTiles(thresh, dil, erode)
    
    
    
    min_w = 50
    min_h = 50
    max_w = 200
    max_h = 200
    rois = 0
    max_iters = 1000
    max_rois = 0
    min_area = 900
    
    # for i in range(min_iters):
    #     rois = 0
    #     (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(erode, 4, cv2.CV_32S)
    #     for i in range(1, num_labels):
    #         w = stats[i, cv2.CC_STAT_WIDTH]
    #         h = stats[i, cv2.CC_STAT_HEIGHT]
    #         area = stats[i, cv2.CC_STAT_AREA]
    #         keep_width = w >= min_w and w <= max_w
    #         keep_height = h >= min_h and w <= max_h
    #         keep_area = area > min_area
    #         if all((keep_width, keep_height, keep_area)):
    #             rois +=1
    #     if rois > max_rois:
    #         max_rois = rois
    #         out = erode.copy()
    #     erode = cv2.erode(erode, np.array([[0,1,0],[1,1,1],[0,1,0]]).astype('uint8'))
    all_rects = []
    seq = []
    for i in range(max_iters):
        if erode.sum() == 0:
            print(i)
            break
        rois = 0
        cnts = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        display = erode.copy()
        for cnt in cnts:
            cv2.drawContours(erode,[cnt],-1,[255,255,255],-1)
            x,y,w,h = cv2.boundingRect(cnt)
            area = w*h
            keep_width = w >= min_w and w <= max_w
            keep_height = h >= min_h and h <= max_h
            keep_area = area > min_area
            if all((keep_width, keep_height, keep_area)):
                rois +=1
                all_rects.append([x,y,w,h])
                cv2.rectangle(display,(x,y),(x+w,y+h),(100,100,100),3)
        # plt.imshow(display)
        # cv2.waitKey(0)        
        if rois > max_rois:
            max_rois = rois
            out = erode.copy()
        erode = cv2.erode(erode, np.array([[0,1,0],[1,1,1],[0,1,0]]).astype('uint8'))
        seq.append(display)
    roi_rects = []
    # if out.any:
    #     (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(out, 4, cv2.CV_32S)
    #     tile_out = tile.copy()
    #     for i in range(1, num_labels):
    #         x = stats[i, cv2.CC_STAT_LEFT]
    #         y = stats[i, cv2.CC_STAT_TOP]
    #         w = stats[i,cv2.CC_STAT_WIDTH]
    #         h = stats[i,cv2.CC_STAT_HEIGHT]
    #         area = stats[i,cv2.CC_STAT_AREA]
    #         center_x, center_y = centroids[i]
    #         keep_width = w >= min_w and w <= max_w
    #         keep_height = h >= min_h and h <= max_h
    #         keep_area = area > min_area
    #         if all((keep_width, keep_height,keep_area)):
    #             roi_rects.append([x,y,w,h])
    #             cv2.rectangle(tile_out, (x,y),(x+w,y+h),(0,255,0),3)
    
    tile_out = tile.copy()
    try:
        cnts = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            keep_width = w >= min_w and w <= max_w
            keep_height = h >= min_h and h <= max_h
            keep_area = area > min_area
            if all((keep_width, keep_height,keep_area)):
                roi_rects.append([x,y,w,h])
                cv2.rectangle(tile_out, (x,y),(x+w,y+h),(0,255,0),3)    
    except:
        out=False
    
    # pltTiles(tile, thresh, tile_out)
    
    
    
    # return roi_rects
    return all_rects

def pltTiles(tile,all_dog,contrast):
    plt.subplot(131)
    plt.imshow(tile)
    plt.subplot(132)
    plt.imshow(all_dog)
    plt.subplot(133)
    plt.imshow(contrast)
    plt.show()


# im = cv2.imread('fl_test.jpg')
# im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
# plt.imshow(im)

name = 'florida_algo_boxes'
bm = cv2.imread('world.200412.3x21600x21600.B1.jpg')
bm_w = bm.shape[1]
bm_h = bm.shape[1]
im_min_lon=-88
im_min_lat =24
im_max_lon =-77
im_max_lat = 31
im_lon_w = im_max_lon - im_min_lon
im_lat_h = im_max_lat - im_min_lat


bm_min_lon = -90
bm_min_lat = 0
bm_lon_w = 90
bm_lat_w = 90

start_x = round(((im_min_lon - bm_min_lon)/bm_lon_w)*bm_w)
start_y = round((1-(im_max_lat - bm_min_lat)/bm_lat_w)*bm_h)
end_x = round(((im_max_lon - bm_min_lon)/bm_lon_w)*bm_w)
end_y = round((1-(im_min_lat - bm_min_lat)/bm_lat_w)*bm_h)

im = bm[start_y:end_y,start_x:end_x]


window_size = 600

im_height = im.shape[0]
im_width = im.shape[1]

hor_tiles = im_width//(window_size)
ver_tiles = im_height//(window_size) 
im_copy = im.copy()
shifted_roi_rects = []
for i in range(hor_tiles):
    for j in range(ver_tiles):
        tile = im[j*window_size:(j*window_size + window_size),i*window_size:(i*window_size + window_size)]
        roi_rects = getROIs(tile) 
        if roi_rects:
            for rect in roi_rects:
                x,y,w,h = rect
                shifted_rect = [x + i*window_size, y + j*window_size, w, h]
                cv2.rectangle(im_copy, (shifted_rect[0],shifted_rect[1]),(shifted_rect[0]+w,shifted_rect[1]+h),[0,255,0],3)
                shifted_roi_rects.append(shifted_rect)
            
all_rects = getROIs(im) + shifted_roi_rects
for rect in all_rects:
    x,y,w,h = rect
    cv2.rectangle(im_copy, (x,y),(x+w,y+h),[0,255,0],3)
plt.imshow(im_copy)
# tuplify
def tup(point):
    return (point[0], point[1]);

# returns true if the two boxes overlap
def overlap(source, target):
    # # unpack points
    tl1, br1 = source;
    tl2, br2 = target;

    # checks
    if (tl1[0] >= br2[0] or tl2[0] >= br1[0]):
        return False;
    if (tl1[1] >= br2[1] or tl2[1] >= br1[1]):
        return False;
    return True;
    # x_a = max(tl1[0],tl2[0])
    # y_a = max(tl1[1],tl2[1])
    # x_b = min(br1[0],br2[0])
    # y_b = min(br1[1],br2[1])
    # inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    # src_area = (br1[0]-tl1[0] + 1) * (br1[1]-tl1[1] + 1)
    # tgt_area = (br2[0]-tl2[0] + 1) * (br2[1] - tl2[1] + 1)
    # iou = inter_area / float(src_area + tgt_area - inter_area)
    # if iou > 0.5:
    #     return True
    # else:
    #     return False

# returns all overlapping boxes
def getAllOverlaps(boxes, bounds, index):
    overlaps = [];
    for a in range(len(boxes)):
        if a != index:
            if overlap(bounds, boxes[a]):
                overlaps.append(a);
    return overlaps;



# go through the contours and save the box edges
boxes = []
for rect in all_rects:
    x,y,w,h = rect
    boxes.append([[x,y],[x+w,y+h]])

# filter out excessively large boxes
filtered = [];
max_area = 300000;
for box in boxes:
    w = box[1][0] - box[0][0];
    h = box[1][1] - box[0][1];
    if w*h < max_area:
        filtered.append(box);
boxes = filtered;

# go through the boxes and start merging
merge_margin = 0

# this is gonna take a long time
finished = False
highlight = [[0,0], [1,1]]
points = [[[0,0]]]
while not finished:
    # set end con
    finished = True

    # check progress
    # print("Len Boxes: " + str(len(boxes)));

    # # draw boxes # comment this section out to run faster
    # copy = np.copy(im);
    # for box in boxes:
    #     cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0,200,0), 1);
    # cv2.rectangle(copy, tup(highlight[0]), tup(highlight[1]), (0,0,255), 2);
    # for point in points:
    #     point = point[0];
    #     cv2.circle(copy, tup(point), 4, (255,0,0), -1);
    # cv2.imshow("Copy", copy);
    # key = cv2.waitKey(1);
    # if key == ord('q'):
    #     break;

    # loop through boxes
    index = len(boxes) - 1
    while index >= 0:
        # grab current box
        curr = boxes[index]

        # add margin
        tl = curr[0][:]
        br = curr[1][:]
        
        merge_margin_x = -(br[0]-tl[0])//5
        merge_margin_y = -(br[1]-tl[1])//5
        
        tl[0] -= merge_margin_x
        tl[1] -= merge_margin_y
        br[0] += merge_margin_x
        br[1] += merge_margin_y

        # get matching boxes
        overlaps = getAllOverlaps(boxes, [tl, br], index)
        
        # check if empty
        if len(overlaps) > 0:
            # combine boxes
            # convert to a contour
            con = [];
            overlaps.append(index);
            for ind in overlaps:
                tl, br = boxes[ind];
                con.append([tl]);
                con.append([br]);
            con = np.array(con);

            # get bounding rect
            x,y,w,h = cv2.boundingRect(con);

            # stop growing
            w -= 1;
            h -= 1;
            merged = [[x,y], [x+w, y+h]];

            # highlights
            highlight = merged[:];
            points = con;

            # remove boxes from list
            overlaps.sort(reverse = True);
            for ind in overlaps:
                del boxes[ind];
            boxes.append(merged);

            # set flag
            finished = False;
            break;

        # increment
        index -= 1;
cv2.destroyAllWindows();

# show final
copy = np.copy(im);
for box in boxes:
    cv2.rectangle(copy, tup(box[0]), tup(box[1]), (0,200,0), 3);
cv2.imshow("Final", copy);
cv2.waitKey(0);

plt.imshow(copy)

out_boxes = []
for box in boxes:
    tl, br = box
    tl_x, tl_y = tl
    br_x, br_y = br
    w = br_x - tl_x
    h = br_y - tl_y
    tl_lon = (tl_x/im_width)*im_lon_w + im_min_lon
    tl_lat = ((1-tl_y/im_height))*im_lat_h + im_min_lat
    br_lon = (br_x/im_width)*im_lon_w + im_min_lon
    br_lat = ((1-br_y/im_height))*im_lat_h + im_min_lat
    # w_lon = (w/im_width)*im_lon_w
    # h_lat = (h/im_height)*im_lat_h
    out_boxes.append([tl_lon,tl_lat,br_lon,br_lat])

with open(name + '.csv','w') as csvfile:
    writer = csv.writer(csvfile,delimiter=',')
    header = ['min_lon','min_lat','max_lon','max_lat']
    writer.writerow(header)
    writer.writerows(out_boxes)
    
       













