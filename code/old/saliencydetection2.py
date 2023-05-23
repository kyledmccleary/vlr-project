import cv2
from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np
import matplotlib.pyplot as plt

# im = cv2.imread('florida_test.jpg')
im = cv2.imread('fl_test.jpg')
im_k = im.reshape((-1,3)).astype('float32')

# edge = cv2.Canny(im, 30, 80)
# # edge = (im.sum(axis=2) != 0).astype('float32')
# dil = binary_dilation(edge,iterations = 20)
# ero = binary_erosion(dil,iterations = 20)
# ero = ero.astype('uint8')*255

max_w = 300
min_w = 50
max_h = 300
min_h = 50

num_regions = 50

# k = 5
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
# km = cv2.kmeans(im_k,k,None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# ret,label,center = km
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((im.shape))
# plt.imshow(res2)
res2 = im

kernel_size = 5
sigma = 0
kernel = (kernel_size, kernel_size)

r = res2[:,:,2].astype('float32')
g = res2[:,:,1].astype('float32')
b = res2[:,:,0].astype('float32')

dog_tiers = 8
gr_list = [r]
gb_list = [b]
gg_list= [g]
dog_list = []


for i in range(dog_tiers):
    gr2 = cv2.GaussianBlur(gr_list[i],kernel,sigma)
    gb2 = cv2.GaussianBlur(gb_list[i],kernel,sigma)
    gg2 = cv2.GaussianBlur(gg_list[i],kernel,sigma)
    dogr = np.abs(gr2-gr_list[i])
    dogr = dogr - dogr.min()
    dogr = dogr/dogr.max()
    dogg = np.abs(gg2-gg_list[i])
    dogg = dogg - dogg.min()
    dogg = dogg/dogg.max()
    dogb = np.abs(gb2-gb_list[i])
    dogb = dogb-dogb.min()
    dogb = dogb/dogb.max()
    dog = dogr + dogg + dogb
    dog_list.append(dog)
    gr = gr2
    gb = gb2
    gg = gg2

    gr_list.append(gr)
    gg_list.append(gg)
    gb_list.append(gb)

cmap_list = []

contrast_filter = np.ones((299,299))*-1 /(299*299)
contrast_filter[149,149] = 1
plt.imshow(contrast_filter)

crs = 0
cbs = 0
cgs = 0

for i in range(len(gg_list)):
    cmap_r = np.zeros(r.shape)
    cmap_g = np.zeros(g.shape)
    cmap_b = np.zeros(b.shape)
    gg = gg_list[i]
    gb = gb_list[i]
    gr = gr_list[i]
    cg = cv2.filter2D(gg,-1,contrast_filter)
    cg = (cg - cg.min())
    cg = cg/cg.max()
    cb = cv2.filter2D(gb,-1,contrast_filter)
    cb = cb-cb.min()
    cb = cb/cb.max()
    cr = cv2.filter2D(gr,-1,contrast_filter)
    cr = cr-cr.min()
    cr = cr/cr.max()
    crs += cr
    cgs += cg
    cbs += cb
cmap = crs + cgs + cbs
cmap_norm = cmap/cmap.max()
dog_all = np.array(dog_list).sum(axis=0)
dog_norm = dog_all/dog_all.max()
combo = dog_norm + cmap_norm
combo_norm = combo/combo.max()
combo_im = np.uint8(combo_norm*255)

val,thresh = cv2.threshold(combo_im,0,255,cv2.THRESH_OTSU)
edges = cv2.Canny(combo_im,50,100)
plt.imshow(thresh)

dil = binary_dilation(thresh,iterations=50)
plt.imshow(dil)

# g1r = cv2.GaussianBlur(r, kernel, sigma)
# g1g = cv2.GaussianBlur(g,kernel,sigma)
# g1b = cv2.GaussianBlur(b,kernel,sigma)

# g2r = cv2.GaussianBlur(g1r, kernel, sigma)
# g2g = cv2.GaussianBlur(g1g, kernel, sigma)
# g2b = cv2.GaussianBlur(g1b, kernel, sigma)

# g3r = cv2.GaussianBlur(g2r, kernel, sigma)

# dogr = np.abs(g2r-g1r)
# dogg = np.abs(g2g-g1g)
# dogb = np.abs(g2b-g1b)

# dog = dogr + dogg + dogb


# g1 = cv2.GaussianBlur(res2,kernel,sigma)
# plt.imshow(g1)
# g2 = cv2.GaussianBlur(g1,kernel,sigma)

# dog = np.abs(g1-g2)
# plt.imshow(dog)

# gray_dog = cv2.cvtColor(dog,cv2.COLOR_RGB2GRAY)
# plt.imshow(gray_dog)
# dil = binary_dilation(gray_dog,iterations=1)

# plt.imshow(dil.astype('uint8')*255)
ero = binary_erosion(dil,iterations=50).astype('uint8')*255
plt.imshow(ero)



mask = np.zeros(ero.shape, dtype='uint8') 


output = im.copy()
cnt = 0
max_cnt = 0

shapes =[]
while(cnt < num_regions):
    cnt = 0
    
    (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(ero, 4,cv2.CV_32S)


    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i,cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        center_x, center_y = centroids[i]
        
        keep_width = w >= min_w and w <= max_w
        keep_height = h >= min_h and h <= max_h
        keep_area = area > 2500
        if all((keep_width, keep_height, keep_area)):
            cnt += 1
            shapes.append([x,y,w,h])
    
    if cnt > max_cnt:
        max_cnt = cnt
        ero_out = ero
    if cnt == 0:
        break
    ero = binary_erosion(ero).astype('uint8')*255
print(max_cnt)    
(num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(ero_out, 4,cv2.CV_32S)    
for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i,cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        center_x, center_y = centroids[i]
        
        keep_width = w >= min_w and w <= max_w
        keep_height = h >= min_h and h <= max_h
        keep_area = area > 2500
        if all((keep_width, keep_height, keep_area)):        
            component_mask = (labels==i).astype('uint8')*255
            mask = cv2.bitwise_or(mask,component_mask)
            cv2.rectangle(output, (x,y),(x+w,y+h),(0,255,0),3)
            cv2.circle(output, (int(center_x),int(center_y)),4, (0,0,255),-1)    
# cv2.imshow('im',im)
# cv2.imshow('edge',edge)
# cv2.waitKey(0)
# cv2.imshow('',ero)

# cv2.imshow('im',im)
# output = im.copy()
# output_arr = np.zeros(ero.shape)
# for shape in shapes:
#     x,y,w,h = shape
#     cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),3)
#     output_arr[x:x+w,y:y+h] += 1
plt.imshow(output)


                
