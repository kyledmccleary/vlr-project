import numpy as np
import os
import cv2
import pyproj

transformer = pyproj.Transformer.from_crs(
    {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
    {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
)
threshold = 0.8
region = '12R'
base_path = './12R_orb_sent'
all_landmarks = np.load('all_landmarks.npy', allow_pickle=True)
av_dist = 0
num_points = 0
for landmark in all_landmarks:
    im_path, points = landmark
    if len(points) > 0:
        npy_path = im_path[:-4] + '.npy'
        lonlat_arr = np.load(os.path.join(base_path,npy_path))
        lonlat_big = cv2.resize(lonlat_arr, (2592,1944), cv2.INTER_CUBIC)
        for point in points:
            x, y, cl, conf = point
            if conf > threshold:
                boxes = np.load(region + '_outboxes.npy')
                minlon, maxlat, maxlon, minlat = boxes[cl]
                midlon = (minlon + maxlon) / 2
                midlat = (minlat + maxlat) / 2
                predict = lonlat_big[y,x]
                xyz_pred = np.array(transformer.transform(predict[0],
                                                          predict[1],0))
                xyz_true = np.array(transformer.transform(midlon, 
                                                          midlat, 0))
                ecef_dist = np.linalg.norm(xyz_true - xyz_pred)
                av_dist += ecef_dist
                num_points += 1
av_dist = av_dist / num_points
print(num_points, av_dist)