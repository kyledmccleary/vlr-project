import numpy as np
import ee
import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
from retry import retry
import shutil
import requests

SENSOR = 'l8'
BATCH_SIZE = 100
max_ims = 1500
SCALE=200

def getMGRS():
    LON_STEP = 6
    LAT_STEP = 8   
    lons = np.arange(-180,180,LON_STEP)
    lats = np.arange(-80,80,LAT_STEP)
    lon_labels = np.arange(1,61)
    lat_labels = ['C','D','E','F','G','H','J','K','L','M',
                  'N','P','Q','R','S','T','U','V','W','X']        
    grid = {}
    for i in range(len(lats)):
        for j in range(len(lons)):
            grid[str(lon_labels[j])+lat_labels[i]] = (lons[j],lats[i],lons[j]+LON_STEP,lats[i]+LAT_STEP)
    
    for i in lon_labels:
        idx = str(i)+'X'
        grid[idx] = (lons[i-1],72,lons[i-1]+LON_STEP,84) 
    grid['31V'] = (0,56,3,64)
    grid['32V'] = (3,56,12,64)
    grid['31X'] = (0,72,9,84)
    grid['33X'] = (9,72,21,84)
    grid['35X'] = (21,72,33,84)
    grid['37X'] = (33,72,42,84)   
    del grid['32X']
    del grid['34X']
    del grid['36X']
    return grid

### grid dict of MGRS, key = column # and row (e.g. '3F'), 
### value = (left, bottom, right, top) 
grid = getMGRS()


alaska = ['3W','4W','3V','4V']
arctic_archipelago = ['10W','11W','12W','13W','14W','15W',
                      '10X','11X','12X','13X','14X','15X']
great_lakes = ['16T','17T']
na_west = ['11S','11R','12R']
caribbean = ['17R','17Q','18Q','19Q']
south_america = ['18H','18G','18F','19F','19G','20G','20H','21H']
italy = ['32S','33S','32T','33T']
russia = ['47X','48X','49X']
japan = ['52S','53S','54S','53T','54T','55T']
scs = ['47M','48M','49M','50M','51M',
       '47N','48N','49N','50N','51N',
       '47P','48P','49P','50P','51P',
       '51Q']


keys = alaska + arctic_archipelago + great_lakes + na_west + caribbean + south_america + italy + russia+ japan + scs

# GRID_SELECTION = 2

# ee.Authenticate(auth_mode = 'notebook')
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')

grid_selection = ['19C','20C','18F','19F','18G','19G','20G','55G','58G',
                  '59G','60G','18H','20H','21H','34H','35H','50H','51H',
                  '52H','53H','54H','55H','56H','60H','19J','22J','33J',
                  '36J','38J','50J','56J','19K','24K','33K','36K','37K',
                  '38K','39K','50K','51K','54K','55K','56K','18L','24L',
                  '25L','33L','37L','38L','39L','50L','51L','52L','53L',
                  '54L','55L','17M','22M','23M','24M','25M','32M','33M',
                  '37M','47M','48M','49M','50M','51M','52M','53M','54M',
                  '55M','56M','17N','21N','22N','29N','30N','31N','32N',
                  '38N','39N','44N','47N','48N','49N','50N','51N','15P',
                  '16P','17P','18P','19P','20P','28P','37P','38P','39P',
                  '43P','44P','47P','48P','49P','50P','51P','4Q','5Q',
                  '13Q','14Q','15Q','16Q','17Q','18Q','19Q','28Q','37Q',
                  '40Q','42Q','43Q','45Q','46Q','48Q','49Q','50Q','51Q',
                  '11R','12R','14R','15R','16R','17R','28R','29R','33R',
                  '34R','35R','36R','37R','39R','40R','41R','42R','50R',
                  '51R','10S','18S','29S','30S','31S','32S','33S','34S',
                  '35S','36S','39S','50S','51S','52S','53S','54S','10T','16T','17T',
                  '18T','19T','20T','29T','30T','31T','32T','33T','34T',
                  '53T','54T','55T','3U','9U','17U','18U','19U','20U',
                  '21U','29U','30U','31U','32U','33U','34U','53U','54U',
                  '57U','3V','4V','5V','6V','7V','8V','9V','10V','11V',
                  '12V','13V','14V','15V','18V','19V','20V','22V','23V',
                  '30V','32V','33V','34V','35V','54V','55V','56V','57V',
                  '58V','59V','60V','1W','2W','3W','4W','5W','6W','7W',
                  '8W','9W','10W','11W','12W','13W','14W','15W','16W',
                  '17W','18W','19W','20W','21W','22W','23W','24W','25W',
                  '26W','27W','28W','33W','34W','35W','36W','37W','38W',
                  '39W','40W','41W','42W','43W','44W','56W','57W','58W',
                  '59W','60W','10X','11X','12X','13X','14X','15X','16X',
                  '17X','18X','19X','20X','21X','22X','23X','24X','25X',
                  '26X','27X','28X','33X','35X','39X','40X','41X','42X',
                  '44X','45X','46X','47X','48X','49X','50X','51X','52X']



###RESTARTING####
#grid_selection = grid_selection[-1:]

grid_selection = ['17R']

rects = {}
# grid_selection = ['16T']
for key in grid_selection:
    left, bottom, right, top = [int(val) for val in grid[key]]
  #  tl = ee.Geometry.Point([int(left),int(top)])
   # br = ee.Geometry.Point([int(bottom),int(right)])
    rect = ee.Geometry.Rectangle([left,top,right,bottom])
    rects[key] = rect

all_filter = ee.Filter.bounds(rects[list(rects.keys())[0]])
region_filters = []
for rect in rects:
    region_filters.append(ee.Filter.bounds(rects[rect]))
    if rect in keys:
        all_filter = ee.Filter.Or(all_filter,ee.Filter.bounds(rects[rect]))
elsweyr_filter = all_filter.Not()
region_filters.append(elsweyr_filter)
i_date = '2020-01-01'
f_date = '2022-12-31'
# date_filter = ee.Filter.date('2022-01-01','2022-12-31')
# date_filter = ee.Filter.Or(date_filter,ee.Filter.date('2022-06-01','2022-06-30'))
date_filter = ee.Filter.date(i_date,f_date)

l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')\
            .filter(date_filter)\
            .filter(ee.Filter.lt('CLOUD_COVER_LAND',50))\
            .select('B4','B3','B2')\
            .sort('DATE_ACQUIRED')
            

def getSentinelURLs(i, key):
    winstart = '-01-01'
    winend = '-03-31'
    sprstart = '-04-01'
    sprend = '-06-30'
    sumstart = '-07-01'
    sumend = '-09-30'
    fallstart = '-10-01'
    fallend = '-12-31'
    
    win22 = ee.Filter.date('2022' + winstart,'2022'+winend)
    win21 = ee.Filter.date('2021' + winstart,'2021'+winend)
    win20 = ee.Filter.date('2020' + winstart,'2020'+winend)
    win19 = ee.Filter.date('2019' + winstart,'2019'+winend)
    win18 = ee.Filter.date('2018' + winstart,'2018'+winend)
    spr22 = ee.Filter.date('2022' + sprstart,'2022'+sprend)
    spr21 = ee.Filter.date('2021' + sprstart,'2021'+sprend)
    spr20 = ee.Filter.date('2020' + sprstart,'2020'+sprend)
    spr19 = ee.Filter.date('2019' + sprstart,'2019'+sprend)
    spr18 = ee.Filter.date('2018' + sprstart,'2018'+sprend)
    sum22 = ee.Filter.date('2022' + sumstart,'2022'+sumend)
    sum21 = ee.Filter.date('2021' + sumstart,'2021'+sumend)
    sum20 = ee.Filter.date('2020' + sumstart,'2020'+sumend)
    sum19 = ee.Filter.date('2019' + sumstart,'2019'+sumend)
    sum18 = ee.Filter.date('2018' + sumstart,'2018'+sumend)
    fal22 = ee.Filter.date('2022' + fallstart,'2022'+fallend)
    fal21 = ee.Filter.date('2021' + fallstart,'2021'+fallend)
    fal20 = ee.Filter.date('2020' + fallstart,'2020'+fallend)
    fal19 = ee.Filter.date('2019' + fallstart,'2019'+fallend)
    fal18 = ee.Filter.date('2018' + fallstart,'2018'+fallend)
    
    win_filter = ee.Filter.Or(ee.Filter.Or(win22,win21),win20)
    win_filter = ee.Filter.Or(ee.Filter.Or(win_filter,win19),win18)
    spr_filter = ee.Filter.Or(ee.Filter.Or(spr22,spr21),spr20)
    spr_filter = ee.Filter.Or(ee.Filter.Or(spr_filter,spr19),spr18)
    sum_filter = ee.Filter.Or(ee.Filter.Or(sum22,sum21),sum20)
    sum_filter = ee.Filter.Or(ee.Filter.Or(sum_filter,sum19),sum18)
    fal_filter = ee.Filter.Or(ee.Filter.Or(fal22,fal21),fal20)
    fal_filter = ee.Filter.Or(ee.Filter.Or(fal_filter,fal19),fal18)
   
    bands = ['B4','B3','B2']
    key_rect = rects[key]
    region_rect = key_rect.buffer(150000).bounds()
    s2 = ee.ImageCollection('COPERNICUS/S2_HARMONIZED').filterBounds(region_rect)\
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20))\
                .select(bands)
    
    
    s2_win = s2.filter(win_filter)               
    s2_spr = s2.filter(spr_filter)    
    s2_sum = s2.filter(sum_filter)               
    s2_fal = s2.filter(fal_filter)
    
 #   if (s2_win.size().getInfo() > 0):
    im_win = s2_win.median().divide(10000).multiply(255).toByte()
    #else:
     #   im_win = ee.Image(1)
    #if (s2_spr.size().getInfo() > 0):
    im_spr = s2_spr.median().divide(10000).multiply(255).toByte()
    #else:
     #   im_spr = ee.Image(1)
    #if (s2_sum.size().getInfo() > 0):
    im_sum = s2_sum.median().divide(10000).multiply(255).toByte()
    #else:
    #    im_sum = ee.Image(1)
    #if (s2_fal.size().getInfo() > 0):
    im_fal = s2_fal.median().divide(10000).multiply(255).toByte()
    #else:
    #    im_fal = ee.Image(1) 
    points = getPointsInRegion(key_rect, 500)   
    for j in range(len(points)): 
        rect = makeRectangle(points[j])
                
        win_url = im_win.getThumbURL({
            'region':rect,
            'format':'png',
            'scale':1000,
            'min':0,
            'max':255,
            'crs':'EPSG:3857'
            })
        spr_url = im_spr.getThumbURL({
             'region':rect,
             'format':'png',
             'scale':1000,
             'min':0,
             'max':255,
             'crs':'EPSG:3857'})   
        sum_url = im_sum.getThumbURL({
            'region':rect,
            'format':'png',
            'scale':1000,
            'min':0,
            'max':255,
            'crs':'EPSG:3857'
            })
        fal_url = im_fal.getThumbURL({
             'region':rect,
             'format':'png',
             'scale':1000,
             'min':0,
             'max':255,
             'crs':'EPSG:3857'})   
        win_name = key + '_win_' + str(j)
        spr_name = key + '_spr_' + str(j)
        sum_name = key + '_sum_' + str(j)
        fal_name = key + '_fal_' + str(j)
        
        print('url',i,'done')
        
        path = os.path.join('mgrs2',key)
        if not os.path.exists(path):
            os.makedirs(path)
            print(path, 'folder created')
        
        r = requests.get(win_url, stream=True)
        if r.status_code !=200:
            r.raise_for_status()
        out_path = os.path.join(path,win_name + '.png')
        with open(out_path, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("done: ", win_name)
        
        r = requests.get(spr_url, stream=True)
        if r.status_code !=200:
            r.raise_for_status()
        out_path = os.path.join(path,spr_name + '.png')
        with open(out_path, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("done: ", spr_name)
        
        r = requests.get(sum_url, stream=True)
        if r.status_code !=200:
            r.raise_for_status()
        out_path = os.path.join(path,sum_name + '.png')
        with open(out_path, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("done: ", sum_name)
        
        r = requests.get(fal_url, stream=True)
        if r.status_code !=200:
            r.raise_for_status()
        out_path = os.path.join(path,fal_name + '.png')
        with open(out_path, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("done: ", fal_name)
    
    
    return# [(win_url,win_name),(spr_url,spr_name), (sum_url,sum_name), (fal_url,fal_name)]

def getPointsInRegion(region,num_points):
    water_land_data = ee.ImageCollection('MODIS/061/MCD12Q1')
    land = water_land_data.select('LW').first()
    mask = land.eq(2)
    points = land.updateMask(mask).stratifiedSample(region=region,scale=1000,
                                                    classBand='LW',numPoints=num_points,geometries=True, seed=np.random.randint(1000000))
    return points.aggregate_array('.geo').getInfo()

def makeRectangle(point):
    point = ee.Geometry.Point(point['coordinates'])
    region = point.buffer(300000).bounds()
    rect = region
    return rect

def getLandsatTiles(i, key):
    winstart = '-01-01'
    winend = '-03-31'
    sprstart = '-04-01'
    sprend = '-06-30'
    sumstart = '-07-01'
    sumend = '-09-30'
    fallstart = '-10-01'
    fallend = '-12-31'
    
    win22 = ee.Filter.date('2022' + winstart,'2022'+winend)
    win21 = ee.Filter.date('2021' + winstart,'2021'+winend)
    win20 = ee.Filter.date('2020' + winstart,'2020'+winend)
    win19 = ee.Filter.date('2019' + winstart,'2019'+winend)
    win18 = ee.Filter.date('2018' + winstart,'2018'+winend)
    spr22 = ee.Filter.date('2022' + sprstart,'2022'+sprend)
    spr21 = ee.Filter.date('2021' + sprstart,'2021'+sprend)
    spr20 = ee.Filter.date('2020' + sprstart,'2020'+sprend)
    spr19 = ee.Filter.date('2019' + sprstart,'2019'+sprend)
    spr18 = ee.Filter.date('2018' + sprstart,'2018'+sprend)
    sum22 = ee.Filter.date('2022' + sumstart,'2022'+sumend)
    sum21 = ee.Filter.date('2021' + sumstart,'2021'+sumend)
    sum20 = ee.Filter.date('2020' + sumstart,'2020'+sumend)
    sum19 = ee.Filter.date('2019' + sumstart,'2019'+sumend)
    sum18 = ee.Filter.date('2018' + sumstart,'2018'+sumend)
    fal22 = ee.Filter.date('2022' + fallstart,'2022'+fallend)
    fal21 = ee.Filter.date('2021' + fallstart,'2021'+fallend)
    fal20 = ee.Filter.date('2020' + fallstart,'2020'+fallend)
    fal19 = ee.Filter.date('2019' + fallstart,'2019'+fallend)
    fal18 = ee.Filter.date('2018' + fallstart,'2018'+fallend)
    
    win_filter = ee.Filter.Or(ee.Filter.Or(win22,win21),win20)
    win_filter = ee.Filter.Or(ee.Filter.Or(win_filter,win19),win18)
    spr_filter = ee.Filter.Or(ee.Filter.Or(spr22,spr21),spr20)
    spr_filter = ee.Filter.Or(ee.Filter.Or(spr_filter,spr19),spr18)
    sum_filter = ee.Filter.Or(ee.Filter.Or(sum22,sum21),sum20)
    sum_filter = ee.Filter.Or(ee.Filter.Or(sum_filter,sum19),sum18)
    fal_filter = ee.Filter.Or(ee.Filter.Or(fal22,fal21),fal20)
    fal_filter = ee.Filter.Or(ee.Filter.Or(fal_filter,fal19),fal18)
   
    bands = ['B4','B3','B2']
    key_rect = rects[key]
    region_rect = key_rect.buffer(300000).bounds()
    num_points = 1000    

    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')\
            .filterBounds(region_rect)\
            .filter(ee.Filter.lt('CLOUD_COVER_LAND',10))\
            .select(bands)
    l8_win = l8.filter(win_filter)
    l8_spr = l8.filter(spr_filter)
    l8_sum = l8.filter(sum_filter)
    l8_fal = l8.filter(fal_filter)
    
    im_win = l8_win.median().multiply(255).toByte()
    im_spr = l8_spr.median().multiply(255).toByte()
    im_sum = l8_sum.median().multiply(255).toByte()
    im_fal = l8_fal.median().multiply(255).toByte()
    points = getPointsInRegion(key_rect, num_points)
    for j in range(len(points)):
        rect = makeRectangle(points[j])
        win_url = im_win.getDownloadURL({
            'region':rect,
            'format':'GeoTIFF',
            'scale':500
            })
        spr_url = im_spr.getDownloadURL({
            'region':rect,
            'format':'GeoTIFF',
            'scale':500
            })
        sum_url = im_sum.getDownloadURL({
            'region':rect,
            'format':'GeoTIFF',
            'scale':500
            })
        fal_url = im_fal.getDownloadURL({
            'region':rect,
            'format':'GeoTIFF',
            'scale':500
            })
        win_name = key + '_win_' + str(j)
        spr_name = key + '_spr_' + str(j)
        sum_name = key + '_sum_' + str(j)
        fal_name = key + '_fal_' + str(j)
        
        print('url',i,'done')
        

def getLandsatURLs(i,key):
    im_list = key[1]
    key = key[0]
    image = ee.Image(im_list.get(i))  
    name = 'l8' + '_' + key + '_' + str(i)
    url = image.getDownloadURL({
      #  'crs':'EPSG:4326',
        'scale':SCALE,
        'format':'GeoTIFF',
        'bands':['B4','B3','B2']})
    print(i, 'done')
    return(url,name,key)
 
@retry(tries = 10, delay=1, backoff=2)
def downloadURLs(index, urls):
    url = urls[0]
    name = urls[1]
    key = urls[2]
    base_path = 'datasets/l8_fl'
    # path = os.path.join(base_path,key)
    path = base_path
    if not os.path.exists(path):
        os.makedirs(path)
        print(path, 'folder created')
    out_path = os.path.join(path,name + '.tif')
    r = requests.get(url, stream=True)
    if r.status_code!=200:
        r.raise_for_status()
    with open(out_path, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("Done: ", index,key,name)
   
@retry(tries = 10, delay=1, backoff=2)
def downloadSentinel(index,urls):
    url = urls[0]
    name = urls[1]
    r = requests.get(url, stream=True)
    if r.status_code !=200:
        r.raise_for_status()
    out_path = name + '.png'
    with open(out_path, 'wb') as out_file:
        shutil.copyfileobj(r.raw, out_file)
    print("done: ", name)

    
if __name__ == '__main__':
    if SENSOR == 'l8':
        for i in range(len(region_filters)):   
            region_filter = region_filters[i]
            
            #####
            rect = ee.Geometry.Rectangle([-87,31,-80,24.5])
            region_filter = ee.Filter.bounds(rect)
            ###
            
            
            grid_choice = grid_selection[i]
            
            #########
            i = len(region_filters)
            
            ###########
            
            l8_filtered = l8.filter(region_filter)
            size = l8_filtered.size().getInfo()
            if size > max_ims:
                count = max_ims
            else:
                count = size
            
            im_list = l8_filtered.toList(count)           
            iterable = enumerate(zip(repeat(grid_choice,count),repeat(im_list,count)))
            #p = Pool(cpu_count())  
            p = Pool(3)
            urls = p.starmap(getLandsatURLs,iterable)
            p.starmap(downloadURLs,enumerate(urls))        
            p.close()
        # num_batches = max_ims / BATCH_SIZE
        # if not num_batches % 1 == 0:
        #     num_batches = int(num_batches) + 1
        # else:
        #     num_batches = int(num_batches)
        # #l8 = l8.filter(region_filters[0])
        # for i in range(num_batches):
        #     im_list = l8.toList(BATCH_SIZE, BATCH_SIZE*i)
        #     iterable = enumerate(zip(repeat('any',BATCH_SIZE),repeat(im_list,BATCH_SIZE)),BATCH_SIZE*i)        
        #     p = Pool(cpu_count())
        #     urls = p.starmap(getLandsatURLs, iterable)
        #     p.starmap(downloadURLs, enumerate(urls))
        #     p.close()
    elif SENSOR == 's2':
        p = Pool(cpu_count())
        # urls = 
        p.starmap(getSentinelURLs,enumerate(grid_selection))
        # print('urls done')
        # new_URL_list = []
        # for url in urls:
        #     new_URL_list = new_URL_list + url
        # np.save('urls.npy',np.array(new_URL_list))
        # # p.starmap(downloadSentinel,enumerate(urls))
        p.close()




