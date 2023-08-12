import argparse
import ee
import requests
from multiprocessing import Pool, cpu_count
import os
import shutil
from retry import retry
import numpy as np
from getMGRS import getMGRS
from itertools import repeat


def get_region_filter_from_bounds(bounds, get_rect=True):
    left, bottom, right, top = bounds
    rect = ee.Geometry.Rectangle([left, bottom, right, top])
    region_filter = ee.Filter.bounds(rect)
    if get_rect:
        return region_filter, rect
    else:
        return region_filter


def get_date_filter(i_date, f_date):
    date_filter = ee.Filter.date(i_date, f_date)
    return date_filter


def get_collection(sensor, region_filter, date_filter, bands=None, cloud_cover_min=0, cloud_cover_max=50):
    if bands is None:
        bands = ['B4', 'B3', 'B2']
    if sensor == 'sentinel':
        collection_string = "COPERNICUS/S2_HARMONIZED"
        collection = ee.ImageCollection(collection_string)
        collection = collection.filter(region_filter).filter(date_filter)
        collection = collection.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_max))
        collection = collection.filter(ee.Filter.gte('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_min))
        collection = collection.select(bands)
    else:
        collection = None
    return collection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bounds', nargs='+', type=int, default=[-84, 24, -78, 32])
    parser.add_argument('-g', '--grid_key', type=str)
    parser.add_argument('-i', '--i_date', type=str, default='2022-01-01')
    parser.add_argument('-f', '--f_date', type=str, default='2022-04-01')
    parser.add_argument('-s', '--scale', type=float, default=200.0)
    parser.add_argument('-m', '--max_ims', type=int, default=5)
    parser.add_argument('-o', '--out_path', type=str, default='tmp')
    parser.add_argument('-r', '--region', type=str, default='tmp')
    parser.add_argument('-e', '--format', type=str, default='GEOTiff')
    parser.add_argument('-si', '--start_index', type=int, default=0)
    parser.add_argument('-c', '--crs', type=str, default=None)
    parser.add_argument('-cc', '--cloud_cover_max', type=float, default=40.0)
    parser.add_argument('-cc_gt', '--cloud_cover_min', type=float, default=0.0)
    parser.add_argument('-ba', '--bands', type=str, nargs='+', default=['B4', 'B3', 'B2'])
    parser.add_argument('-se', '--sensor', type=str, default='sentinel')
    parser.add_argument('-bu', '--buffer', type=int, default=250000)
    parser.add_argument('-ll', '--lonlat', type=bool, default=False)
    args = parser.parse_args()

    if args.grid_key is not None:
        grid = getMGRS()
        left, bottom, right, top = grid[args.grid_key]
        args.bounds = [float(left), float(bottom), float(right), float(top)]

    return args


def get_points_in_region(region, args):
    water_land_data = ee.ImageCollection('MODIS/061/MCD12Q1')
    land = water_land_data.select('LW').first()
    mask = land.eq(2)
    points = land.updateMask(mask).stratifiedSample(region=region,
                                                    scale=args.scale,
                                                    classBand='LW',
                                                    numPoints=args.max_ims,
                                                    geometries=True,
                                                    seed=np.random.randint(100000))
    return points.aggregate_array('.geo').getInfo()


def make_rectangle(point, args):
    point = ee.Geometry.Point(point['coordinates'])
    point_rect = point.buffer(args.buffer).bounds()
    return point_rect


def get_url(index, data):
    region_point, args, im = data
    region = make_rectangle(region_point, args)
    im = im.clip(region)
    if args.crs is not None:
        crs = args.crs
    else:
        crs = im.select(0).projection()
    url = im.getDownloadUrl({
        'scale': args.scale,
        'format': args.format,
        'crs': crs,
        'region': region
    })
    print('URL', index, 'done')
    return url

def get_urls_pnglonlat(index, data):
    region_point, args, im, lonlat_im = data
    region = make_rectangle(region_point, args)
    im = im.clip(region)
    lonlat_im = lonlat_im.clip(region)
    if args.crs is not None:
        crs = args.crs
    else:
        crs = im.select(0).projection()
    pic_url = im.getThumbUrl({
        'scale': args.scale,
        'format': args.format,
        'crs': crs,
        'region': region
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
def download_url(index, iterable):
    args, url = iterable
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        print(args.out_path, 'folder created')
    if args.format == 'GEOTiff':
        ext = '.tif'
    elif args.format == 'jpg':
        ext = '.jpg'
    else:
        ext = '.png'
    out_name = args.sensor + '_' + args.region + '_' + str(index).zfill(5) + ext
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        r.raise_for_status()
    with open(os.path.join(args.out_path, out_name), 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print('Download', out_name, 'done')

@retry(tries=10, delay=1, backoff=2)
def download_urls_lonlat(index, iterable):
    args, urls = iterable
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
    


ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def main():
    args = parse_args()
    date_filter = get_date_filter(args.i_date, args.f_date)
    region_filter, region_rect = get_region_filter_from_bounds(args.bounds, get_rect=True)
    collection = get_collection(args.sensor, region_filter, date_filter, bands=None,
                                cloud_cover_min=args.cloud_cover_min, cloud_cover_max=args.cloud_cover_max)
    region_points = get_points_in_region(region_rect, args)
    region_im = collection.mosaic().divide(10000).multiply(255).toByte()
    lonlat_im = ee.Image.pixelLonLat()
    # rect_list = []
    # for point in region_points:
    #     im_rect = make_rectangle(point, args)
    #     rect_list.append(im_rect)
    p = Pool(cpu_count())
    if(args.lonlat):
        iterable = enumerate(zip(region_points, repeat(args), repeat(region_im), repeat(lonlat_im)), args.start_index)
        urls = p.starmap(get_urls_pnglonlat, iterable)
        iterable2 = enumerate(zip(repeat(args), urls), args.start_index)
        p.starmap(download_urls_lonlat, iterable2)
    else:
        iterable = enumerate(zip(region_points, repeat(args), repeat(region_im)), args.start_index)   
        urls = p.starmap(get_url, iterable)
        iterable2 = enumerate(zip(repeat(args), urls), args.start_index)
        p.starmap(download_url, iterable2)
    p.close()
    p.join()


if __name__ == '__main__':
    main()
