from im_pipe import im_pipe
import time
import numpy as np
import cv2
from PIL import Image
from od_pipe import od_pipe

def main():
    
    BO_detection_thresh = 100
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)    
    
    all_dets = []
    tot_detections = 0
        
    # Loop camera capture
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        capture_time = time.time()
    
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame. Exiting")
            break

        # convert cv2 BGR array to RGB PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_im = Image.fromarray(frame_rgb).convert('RGB')
        
        
        # run image pipeline on image
        keys, detections = im_pipe(frame_im)
        tot_detections += len(detections)
        all_dets.append([capture_time, detections])
        print('regions: ', keys, '\t num dets: ', len(detections))
        for landmark in detections:
            x, y, lonlat, conf = landmark
            lonlat_str = '(' + str(round(lonlat[0],4)) + ',' + str(round(lonlat[1],4)) + ')'
            cv2.circle(frame, (x,y), 5,
                        [0,0,255], thickness=-1)
            cv2.putText(frame,
                        lonlat_str + " " + str(round(conf,2)),
                        (x, y), 0, 0.5, [0,0,255])
        s = ''.join(str(key) + ' ' for key in keys)
        cv2.putText(frame,
                    s,
                    (1000, 50), 0, 1, [255, 200, 255], thickness=2)
        
        # display frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
         
        ##  send to OD  ##
        if tot_detections >= BO_detection_thresh:
            # results = od_pipe(all_dets)
            tot_detections = 0
            all_dets = []

    cap.release()
    cv2.destroyAllWindows()
       
if __name__ == '__main__':
    main()