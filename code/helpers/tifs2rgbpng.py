import rasterio
import cv2
import os
from tqdm import tqdm

path = 'modis_images/val/17R'
files = os.listdir(path)

for file in tqdm(files):
    if file.endswith('tif'):
        with rasterio.open(os.path.join(path,file)) as src:
            im = src.read()
            im = im.transpose((1,2,0))
            im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            outpath = os.path.join(path,file[:-4] + '.jpg')
            cv2.imwrite(outpath,im_bgr)
        os.remove(os.path.join(path,file))