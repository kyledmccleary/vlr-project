# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:13:18 2023

@author: kmccl
"""
import os
import csv
import argparse
import numpy as np
from multiprocess import Pool, cpu_count
from io import StringIO

BASEPATH = '.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=True, type=str)
    parser.add_argument('-r', '--resize', default=True, type=bool)
    args = parser.parse_args()
    return args


def get_csvs(path):
    filename_list = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            filename_list.append(file)
    return filename_list


def csv2array(filepath):
    converted=StringIO()
    with open(filepath, 'r', newline=None) as file:
        converted.write(file.read().replace('], ', '\n').replace('[',''))
    converted.seek(0)
    converted.getvalue()
    reader = csv.reader(converted, delimiter =',')
    val_list = []
    for row in reader:
        u = int(row[0][1:])
        v = int(row[1][:-1])
        lon = float(row[3][:-1])
        lat = float(row[2][2:])
        val_list.append([u,v,lon,lat])
    # val_array = np.array(val_list)
    # maxes = val_array.max(axis=0)
    array_width = 216#maxes[0]
    array_height = 162#maxes[1]
    lonlat_array = np.zeros((array_height, array_width,2))
    for item in val_list:
        x, y, lon, lat = item
        if x<array_width and y<array_height:
            lonlat_array[y,x,0] = lon
            lonlat_array[y,x,1] = lat
    print(filepath, 'done')
    return lonlat_array
    

def main():
    args = parse_args()
    path = os.path.join(BASEPATH,args.dir)
    filename_list = get_csvs(path)  
    filepath_list = []
    for filename in filename_list:
        filepath_list.append(os.path.join(path,filename))
    p = Pool(cpu_count())       
    lonlat_array = p.map(csv2array,filepath_list)
    #lonlat_array = csv2array(filepath)
    p.close()
    p.join()
    for i in range(len(filename_list)):
        filename = filename_list[i]
        filepath = filepath_list[i]
        new_filename = filename[:-3] + 'npy'
        new_filepath = os.path.join(path,new_filename)
        np.save(new_filepath,lonlat_array[i])  
        os.remove(filepath)

if __name__ == '__main__':
    main()