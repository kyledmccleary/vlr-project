from im_pipe import im_pipe
import time






def main():
    
    #LOOP
    ## GET IMAGE
    samp_im = 'samp_im.png'
    capture_time = time.time()
    
    
    ## RUN IMAGE PIPELINE ON IMAGE
    keys, detections = im_pipe(samp_im)
    
    ## send to OD
    output = od_pipe(capture_time, detections)
    
if __name__ == '__main__':
    main()