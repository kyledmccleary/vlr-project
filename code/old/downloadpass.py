import ee
import csv
from tqdm import tqdm
import numpy as np
from retry import retry
import os
import requests
import shutil
from multiprocessing import Pool

out_path = 'sequence2'
out_format = None


@retry(tries=10, delay=1, backoff=2)
def downloadURL(index, url):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(out_path, 'folder created')
    if out_format == 'GEOTiff':
        ext = '.tif'
    else:
        ext = '.png'
    out_name = 'ISSGT_' + str(index).zfill(5) + ext
    r = requests.get(url, stream=True)
    if r.status_code !=200:
        r.raise_for_status()
    with open(os.path.join(out_path,out_name),'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print('Download',out_name, 'done')


    

lons = []
lats = []
with open('lonlat.csv') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        if row:
            lons.append(float(row[0]))
            lats.append(float(row[1]))
            
ee.Initialize()
lonlats = np.vstack((lons,lats)).T
rect = ee.Geometry.Rectangle([7,44,19,36])
coll = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA').filterDate('2022-10-01','2023-3-28').filterBounds(rect).filter(ee.Filter.lt('CLOUD_COVER_LAND',10)).select('B4','B3','B2')
im = coll.mosaic().multiply(255).toByte()

imrects = []
for i in tqdm(range(len(lonlats))):
    point = ee.Geometry.Point([lonlats[i][0],lonlats[i][1]])
    imrect = point.buffer(150000).bounds()
    imrects.append(imrect)

def getURL(index, rect):
     url = im.getThumbURL({
        'region':rect,
        'scale':500,
        'format':'png'})
     print(index,'url done')
     return url


if __name__ == '__main__':
    
    p = Pool(24)
    urls = p.starmap(getURL,enumerate(imrects))
    p.starmap(downloadURL,enumerate(urls))
    p.close()
    p.join()
    
   
    
    
    
