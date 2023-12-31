import argparse
import ee
import requests
from multiprocessing import Pool,cpu_count
import os
import shutil
from retry import retry
import numpy as np
from getMGRS import getMGRS


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
    
def getCollection(landsat,region_filter,date_filter,bands=['B4','B3','B2'], cloud_cover_max = 50, date_sort=True):
    collection_string = 'LANDSAT/LC0' + landsat + '/C02/T1_TOA'
    collection = ee.ImageCollection(collection_string)
    collection = collection.filter(region_filter).filter(date_filter)
    collection = collection.filter(ee.Filter.lt('CLOUD_COVER_LAND',cloud_cover_max))
    collection = collection.select(bands)
    if date_sort:
        collection = collection.sort('DATE_ACQUIRED')       
    return collection




ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
parser = argparse.ArgumentParser(
            prog='LandsatDownloader',
            description='Download Landsat imagery from Google Earth Engine')
parser.add_argument('-b', '--bounds', nargs='+', type=int, default=[-84, 24, -78, 32])
parser.add_argument('-g', '--grid_key', type=str)
parser.add_argument('-i', '--idate',type=str)
parser.add_argument('-f', '--fdate',type=str)
parser.add_argument('-s', '--scale', type = float, default = 200.0)
parser.add_argument('-m', '--maxims', type = int, default = 100)
parser.add_argument('-l', '--landsat', choices=['8','9'], type=str, default = '8')    
parser.add_argument('-o', '--outpath', type=str, default = 'landsat_images')   
parser.add_argument('-r', '--region', type=str, default=None)
parser.add_argument('-e', '--format', type=str,default = 'GEOTiff')
parser.add_argument('-mo', '--mosaic',type=bool, default=False)
parser.add_argument('-si', '--startindex',type=int,default=0)
parser.add_argument('-se', '--seed', type=int,default = None)
parser.add_argument('-gm', '--getmonthlies', type=bool, default = False)
parser.add_argument('-c', '--crs', type=str, default = None)
<<<<<<< HEAD:code/helpers/landsatdownloader.py
parser.add_argument('-cc', '--cloud_cover_max',type=float, default = 50.0)
parser.add_argument('-ccgt', '--cloud_cover_min', type=float, default = 0.0)
parser.add_argument('-ba','--bands',type=str,nargs='+',default =['B4','B3','B2'])
parser.add_argument('-ll', '--lonlat', type=bool, default=False)
=======
>>>>>>> c423822f24984b7a76988adc8e4eb9df8b3bbee8:code/landsatdownloader.py

args = parser.parse_args()      
scale = args.scale
max_ims = args.maxims
region_name = args.region
out_path = args.outpath
out_format = args.format

if args.grid_key is not None:
       grid = getMGRS()
       left, bottom, right, top = grid[args.grid_key]
       args.bounds = [float(left), float(bottom), float(right), float(top)]
if args.region is None:
    args.region = args.grid_key
if args.seed:
    seed = args.seed
else:
    seed = np.random.randint(100000)


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
    region = point.buffer(200000).bounds()
    rect = region
    return rect


if not args.mosaic:
    collection = getCollection(args.landsat, region_filter, date_filter,
                           bands=bands, cloud_cover_max = 50, date_sort=True)
    collection_size = collection.size().getInfo()
    if collection_size < max_ims:
        max_ims = collection_size
    im_list = collection.toList(max_ims)  

else:
    im_list = []
    if args.getmonthlies:
        days = ['31','28','31','30','31','30','31','31','30','31','30','31']
        for i in range(1,13):
            if args.seed:
                seed = args.seed
            else:
                seed = np.random.randint(100000)
            idate = args.idate + '-' + str(i) + '-01'
            fdate = args.idate + '-' + str(i) + '-' + days[i-1]
            date_filter = ee.Filter.date(idate,fdate)
            collection = getCollection(args.landsat, region_filter, date_filter,
                                       bands=bands, cloud_cover_max=50, date_sort=False)
            points = getPointsInRegion(rect, max_ims,seed)
            im = collection.mosaic().multiply(255).toByte() 
            for point in points:
                rect = makeRectangle(point)
                rect_im = im.clip(rect)
                im_list.append(rect_im)
    else:
        if args.seed:
            seed = args.seed
        else:
            seed = np.random.randint(100000)
        collection = getCollection(args.landsat, region_filter, date_filter,
                                       bands=bands, cloud_cover_max=50, date_sort=False)
        points = getPointsInRegion(rect, max_ims,seed)
        im = collection.mosaic().multiply(255).toByte() 
        for point in points:
            rect = makeRectangle(point)
            rect_im = im.clip(rect)
            im_list.append(rect_im)
    im_list = ee.List(im_list)
    

def getURL(index):
    image = ee.Image(im_list.get(index))
    if args.crs:
        crs = args.crs
    else:
        crs = image.select(0).projection()
    url = image.getDownloadURL({
        'scale':scale,
        'format':out_format,
        'bands':bands,
        'crs':crs})
    print('URL',index,'done')
    return url

def get_urls_pnglonlat(index):
    image = ee.Image(im_list.get(index))
    image = image.multiply(255).toByte()
    lonlat_im = ee.Image.pixelLonLat()
    region = image.geometry()
    if args.crs is not None:
        crs = args.crs
    else:
        crs = image.select(0).projection()
    pic_url = image.getThumbUrl({
        'scale': args.scale,
        'format': args.format,
        'bands':bands
    })
    lonlat_url = lonlat_im.getDownloadUrl({
        'scale': args.scale*10,
        'format': 'npy',
        'crs': crs,
        'region': region
    })
    print('URLs', index, 'done')
    return [pic_url, lonlat_url]

@retry(tries=10, delay=1, backoff=2)
def download_urls_lonlat(index, urls):
    pic_url, lonlat_url = urls
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print(args.out_path, 'folder created')
    if args.format == 'GEOTiff':
        ext = '.tif'
    elif args.format == 'jpg':
        ext = '.jpg'
    else:
        ext = '.png'
    out_name_pic = args.sensor + '_' + args.region + '_' + str(index).zfill(5) + ext
    out_name_latlon = args.sensor + '_' + args.region + '_' + str(index).zfill(5) + '.npy'
    r = requests.get(pic_url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()
    with open(os.path.join(args.out_path, out_name_pic), 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    r = requests.get(lonlat_url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()
    with open(os.path.join(args.out_path, out_name_latlon), 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print('Download', out_name_pic, 'done')

@retry(tries=10, delay=1, backoff=2)
def downloadURL(index, url):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(out_path, 'folder created')
    if out_format == 'GEOTiff':
        ext = '.tif'
    else:
        ext = '.png'
    out_name = 'l' + args.landsat + '_' + region_name + '_' + str(index).zfill(5) + ext
    r = requests.get(url, stream=True)
    if r.status_code !=200:
        r.raise_for_status()
    with open(os.path.join(out_path,out_name),'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print('Download',out_name, 'done')

def main():
    if args.getmonthlies:
        indexes = range(args.startindex,args.startindex+max_ims*12)
    else:
        indexes = range(args.startindex,args.startindex+max_ims)
    p = Pool(cpu_count())
    if(args.lonlat):
        urls = p.starmap(get_urls_pnglonlat, zip(indexes))
        p.starmap(download_urls_lonlat, enumerate(urls, args.startindex))
    else:
        urls = p.starmap(getURL,zip(indexes))
        p.starmap(downloadURL,enumerate(urls))
    p.close()
    p.join()
    
    
if __name__ == '__main__':
    main()
    