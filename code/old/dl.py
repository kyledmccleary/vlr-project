import numpy as np
import ee
import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
from retry import retry
import shutil
import requests
from getMGRS import getMGRS


BUFFER = 300000
PATH = '../data3/modis3'
START_INDEX = 0
SCALE = 1000
NUM_IMS = 1000
KEY = '17R'
CLOUD_VAL = 50

SENSOR = 'LANDSAT'
# SENSOR = 'MODIS_BIG'
# SENSOR = 'LONLAT'

I_DATE = '2023-03-01'
F_DATE = '2023-03-18'

ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

def getLandsatCollection(date_filter,region_filter,bands):
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')\
                .filter(date_filter)\
                .filter(region_filter)\
                .filter(ee.Filter.lt('CLOUD_COVER_LAND',CLOUD_VAL))\
                .select(bands)\
                .sort('CLOUD_COVER_LAND')   
    return l8

def getSeasonFilters(initial_year, final_year):
    initial_year_int = int(initial_year)
    initial_year_str = str(initial_year)
    final_year_int = int(final_year)
    
    win_start = '-01-01'
    win_end = '-03-31'
    spr_start = '-04-01'
    spr_end = '-06-30'
    sum_start = '-07-01'
    sum_end = '-09-30'
    fal_start = '-10-01'
    fal_end = '-12-31'
    
    win_filter = ee.Filter.date(initial_year_str + win_start,initial_year_str + win_end)
    spr_filter = ee.Filter.date(initial_year_str + spr_start,initial_year_str + spr_end)
    sum_filter = ee.Filter.date(initial_year_str + sum_start,initial_year_str + sum_end)
    fal_filter = ee.Filter.date(initial_year_str + fal_start,initial_year_str + fal_end)
    
    for i in range(initial_year_int+1,final_year_int+1):
        win_filter = ee.Filter.Or(win_filter,ee.Filter.date(str(i) + win_start,str(i) + win_end))
        spr_filter = ee.Filter.Or(spr_filter,ee.Filter.date(str(i) + spr_start,str(i) + spr_end))
        sum_filter = ee.Filter.Or(sum_filter,ee.Filter.date(str(i) + sum_start,str(i) + sum_end))
        fal_filter = ee.Filter.Or(fal_filter,ee.Filter.date(str(i) + fal_start,str(i) + fal_end))
    return win_filter, spr_filter, sum_filter, fal_filter

def getPointsInRegion(region_geometry,num_points,scale,seed=None):
    if not seed:
        seed = np.random.randint(100000)
    water_land_data = ee.ImageCollection('MODIS/061/MCD12Q1')
    land = water_land_data.select('LW').first()
    mask = land.eq(2)
    points = land.updateMask(mask).stratifiedSample(region=region_geometry,
                                                    scale = scale,
                                                    classBand = 'LW',
                                                    numPoints = num_points,
                                                    geometries = True,
                                                    seed = seed)
    return points.aggregate_array('.geo').getInfo() 

def makeSquareFromPoint(point,buffer = BUFFER):
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(buffer).bounds()
    rect = region
    return rect

def getMODISCollection(date_filter, bands):
    modis = ee.ImageCollection("MODIS/061/MOD09GA")\
                .filter(date_filter)\
                .select(bands)
    return modis

def getURL(i,iterable):
    point = iterable[0]
    collection = iterable[1]
    rect = makeSquareFromPoint(point)
    col_filtered = collection.filterBounds(rect)
    im = col_filtered.median().multiply(255).toByte()
    url = im.getDownloadURL({
        'region':rect,
        'format':'GeoTIFF',
        'scale':SCALE,
        'crs':col_filtered.select(0).first().projection()
        })
    print('url',i,'done')
    return url

def getMODISURL(i,iterable):
    point = iterable[0]
    image = iterable[1]
    rect = makeSquareFromPoint(point)
    image = image.divide(10000).multiply(255).toByte()
    
    url = image.getDownloadURL({
        'region':rect,
        'format':'GeoTIFF',
        'scale':SCALE,
        'crs':'EPSG:3857'})
    print('url',i,'done')
    return url

def getBigMODISURL(i, iterable):
    image = iterable[0]
    rect = iterable[1]
    image = image.divide(10000).multiply(255).toByte()
    
    #grayscale via luminance
    # image = image.expression('(0.3*R) + (0.59*G) + (0.11*B)',{
    #     'R': image.select('sur_refl_b01'),
    #     'G': image.select('sur_refl_b04'),
    #     'B': image.select('sur_refl_b03')})
    # image = image.toByte()
    
    url = image.getThumbURL({
        'region':rect,
        #'format':'GeoTIFF',
        'format':'png',
        'scale':SCALE,
        # 'crs':image.select(0).projection()
        'crs':'EPSG:3857'
        })
    print('url',i,'done')
    return url

@retry(tries = 10, delay=1, backoff=2)    
def downloadURL(i, url):
    path = PATH
    if not os.path.exists(PATH):
        os.makedirs(path)
        print(path, 'folder created')
    r = requests.get(url, stream=True)
    if r.status_code !=200:
        r.raise_for_status()
    if SENSOR == 'MODIS':
        out_path = os.path.join(PATH,'MODIS_' + str(i).zfill(5) + '.tif')
    elif SENSOR == 'MODIS_BIG':
        out_path = os.path.join(PATH,'MODIS_' + str(i).zfill(5) + '.png')
    else:
        out_path = os.path.join(PATH,str(i).zfill(5) + '.tif')
    with open(out_path, 'wb') as out_file:
        shutil.copyfileobj(r.raw,out_file)
    print('download',i,'done')


                  

if __name__ == '__main__':    
    if SENSOR == 'LANDSAT':
        initial_year = I_DATE
        final_year = F_DATE 
        win_filter, spr_filter, sum_filter, fal_filter = getSeasonFilters(initial_year, final_year)   
        bands = ['B4','B3','B2']
        buffer = BUFFER
        key = KEY
        grid = getMGRS()    
        left, bottom, right, top = [int(val) for val in grid[key]]  
        region_geometry = ee.Geometry.Rectangle([left,top,right,bottom])
        buffered_region_geometry = region_geometry.buffer(buffer).bounds()
        region_filter = ee.Filter.bounds(buffered_region_geometry)
         
        l8_win = getLandsatCollection(win_filter, region_filter, bands)    
        l8_spr = getLandsatCollection(spr_filter, region_filter, bands)  
        l8_sum = getLandsatCollection(sum_filter, region_filter, bands)  
        l8_fal = getLandsatCollection(fal_filter, region_filter, bands)  
        
        num_points = NUM_IMS
        scale = SCALE
        seed = 0
        
        points = getPointsInRegion(region_geometry,num_points,scale,seed)
        points_out = [point['coordinates'] for point in points]
        points_out_arr = np.array(points_out)
        np.save('points',points_out_arr)
        collection = np.random.choice([l8_win,l8_spr,l8_sum,l8_fal],num_points)
        iterable = enumerate(zip(points,collection),START_INDEX)
        
        p = Pool(cpu_count())
        urls = p.starmap(getURL,iterable)
        p.starmap(downloadURL,enumerate(urls,START_INDEX))
        p.close()
    elif SENSOR == 'MODIS':
        i_year = '2018'
        f_year = '2023'
        bands = ['sur_refl_b01','sur_refl_b04','sur_refl_b03']
        date_filter = ee.Filter.date(i_year,f_year)
        modis = getMODISCollection(date_filter,bands)
        buffer = BUFFER
        key = KEY
        grid = getMGRS()
        left, bottom, right, top = [int(val) for val in grid[key]]
        region_geometry = ee.Geometry.Rectangle([left,top,right,bottom])
        collection_size = modis.size().getInfo()
        im_list = modis.toList(collection_size)
        num_points = 5
        scale = SCALE
        im_list2 = []
        pt_list = []
        for i in range(collection_size):
            seed = np.random.randint(100000)
            im = ee.Image(im_list.get(i))
            points = getPointsInRegion(region_geometry,num_points,scale,seed)
            ims = [im]*num_points
            im_list2 = im_list2 + ims
            pt_list = pt_list + points
        iterable = enumerate(zip(pt_list,im_list2),START_INDEX)
        p = Pool(cpu_count())
        urls = p.starmap(getMODISURL,iterable)
        p.starmap(downloadURL,enumerate(urls,START_INDEX))
        p.close()
    elif SENSOR == 'MODIS_BIG':
        i_year = I_DATE
        f_year = F_DATE
        bands = ['sur_refl_b01','sur_refl_b04','sur_refl_b03']
        date_filter = ee.Filter.date(i_year,f_year)
        modis = getMODISCollection(date_filter,bands)
        key = KEY
        grid = getMGRS()
        left, bottom, right, top = [int(val) for val in grid[key]]
        region_geometry = ee.Geometry.Rectangle([left,top,right,bottom]).buffer(300000).bounds()
        collection_size = modis.size().getInfo()
        im_list = modis.toList(collection_size)
        scale = SCALE
        im_list2 = []
        rect_list = []
        for i in range(collection_size):
            im_list2.append( ee.Image(im_list.get(i)))
            rect_list.append(region_geometry)
        iterable = enumerate(zip(im_list2, rect_list),START_INDEX)
        p = Pool(cpu_count())
        urls = p.starmap(getBigMODISURL,iterable)
        p.starmap(downloadURL,enumerate(urls,START_INDEX))
        p.close()
    elif SENSOR == 'LONLAT':
        im = ee.Image.pixelLonLat()
        key = KEY
        grid = getMGRS()
        left, bottom, right, top = [int(val) for val in grid[key]]
        region_geometry = ee.Geometry.Rectangle([left,top,right,bottom]).buffer(300000).bounds()
        scale = SCALE
        url = im.getDownloadURL({
            'format':'npy',
            'crs':'EPSG:3857',
            'scale':scale,
            'region':region_geometry})
        print(url)
        
        
        
        
        