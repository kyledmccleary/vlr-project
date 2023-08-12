import argparse
import ee
import requests
from multiprocessing import Pool,cpu_count
import os
import shutil
from retry import retry
import numpy as np
from itertools import repeat
from tqdm import tqdm


def getRegionFilterFromBounds(bounds, getRect=True):
    left, top, right, bottom = bounds
    rect = ee.Geometry.Rectangle([left,top,right,bottom])
    region_filter = ee.Filter.bounds(rect)
    if getRect:
        return region_filter, rect
    else:
        return region_filter
  
def getDateFilter(i_date, f_date):
    date_filter = ee.Filter.date(i_date,opt_end=f_date)
    return date_filter
    
def getCollection(region_filter,date_filter,bands=['B4','B3','B2'], cloud_cover_max = 50, date_sort=True):
    collection_string = 'LANDSAT/LC08/C02/T1_TOA'
    collection = ee.ImageCollection(collection_string)
    collection = collection.filter(region_filter).filter(date_filter)
    collection = collection.filter(ee.Filter.lt('CLOUD_COVER_LAND',cloud_cover_max))
    collection = collection.select(bands)
    if date_sort:
        collection = collection.sort('DATE_ACQUIRED')       
    return collection




ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
parser = argparse.ArgumentParser(
            prog='Landsat8MonthlyMosaicDownloader',
            description='Download monthly mosaiced Landsat8 imagery from Google Earth Engine')
parser.add_argument('-b', '--bounds', nargs='+', type=int)
parser.add_argument('-i', '--idate',type=str)
parser.add_argument('-f', '--fdate',type=str)
parser.add_argument('-s', '--scale', type = float, default = 200.0)
parser.add_argument('-m', '--monthlyims', type = int, default = 20)
parser.add_argument('-o', '--outpath', type=str, default = 'landsat_images')   
parser.add_argument('-n', '--name', type=str)
parser.add_argument('-e', '--format', type=str,default = 'GEOTiff')
parser.add_argument('-si', '--startindex',type=int,default=0)
parser.add_argument('-se', '--seed', type=int,default = None)
parser.add_argument('-c', '--crs', type=str, default = 'EPSG:3857')
parser.add_argument('-bu', '--buffer', type=int, default = 100000)

args = parser.parse_args()      
scale = args.scale
region_name = args.name
out_path = args.outpath
out_format = args.format
batch_size = 500

bands = ['B4','B3','B2']
region_filter, rect = getRegionFilterFromBounds(args.bounds,getRect=True)
date_filter = getDateFilter(args.idate, args.fdate)


def getPointsInRegion(region, num_points, seed):
    water_land_data = ee.ImageCollection('MODIS/061/MCD12Q1')
    land = water_land_data.select('LW').first()
    mask = land.eq(2)
    points = land.updateMask(mask).stratifiedSample(region=region, scale = scale,
                                                    classBand = 'LW', numPoints = num_points,
                                                    geometries=True,seed = seed)
    return points.aggregate_array('.geo').getInfo()

def makeRectangle(point):
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(args.buffer).bounds()
    rect = region
    return rect

def genImList():
    im_list = []
    for i in tqdm(range(1,13)):
        if args.seed:
            seed = args.seed
        else:
            seed = np.random.randint(100000)
        idate = args.idate + '-' + str(i)
        if i ==12:
            fdate = args.fdate
        else:
            fdate = args.idate + '-' + str(i+1)
        date_filter = ee.Filter.date(idate,fdate)
        collection = getCollection(region_filter, date_filter,bands = bands, cloud_cover_max = 100, date_sort = False)
        im = collection.mosaic().multiply(255).toByte()
        points = getPointsInRegion(rect, args.monthlyims, seed)
        for point in points:
            minirect = makeRectangle(point)
            rect_im = im.clip(minirect)
            im_list.append(rect_im)
    num_ims = len(im_list)
    im_list = ee.List(im_list)
    return im_list, num_ims
    

def getURL(index, im_list):
    image = ee.Image(im_list.get(index-args.startindex))
    url = image.getDownloadURL({
        'scale':scale,
        'format':out_format,
        'crs':args.crs})
    print('URL',index,'done')
    return url

@retry(tries=10, delay=1, backoff=2)
def downloadURL(index, url):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(out_path, 'folder created')
    if out_format == 'GEOTiff':
        ext = '.tif'
    else:
        ext = '.png'
    out_name = 'l8_' + region_name + '_' + str(index).zfill(5) + ext
    r = requests.get(url, stream=True)
    if r.status_code !=200:
        r.raise_for_status()
    with open(os.path.join(out_path,out_name),'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print('Download',out_name, 'done')

def main():
    im_list, num_ims = genImList()
    batches = num_ims//batch_size + 1
    lastbatch = num_ims - (batches-1) * batch_size
    p = Pool(cpu_count())
    for i in range(batches):
        if i == batches-1:
            im_ct = lastbatch
        else:
            im_ct = batch_size
        indexes = range(args.startindex+i*batch_size, args.startindex+im_ct + i*batch_size)
        urls = p.starmap(getURL,zip(indexes,repeat(im_list)))
        p.starmap(downloadURL,enumerate(urls,i*batch_size+args.startindex))
    p.close()
    p.join()
    
    
if __name__ == '__main__':
    main()
    