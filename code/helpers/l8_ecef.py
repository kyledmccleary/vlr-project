import csv
import numpy as np


with open('LC08_L1TP_017039_20230314_20230321_02_T1_ANG.txt') as infile:
    file = infile.readlines()
    num_points = file[25].strip()[-2:]
    num_points = int(num_points)
    start_time = file[24][-13:]
    start_time = float(start_time)
    end_time = start_time + num_points
    ecef_x = file[37:48]
    ecef_x_list = []
    for row in ecef_x:
        cols = row.split()
        for col in cols:
            try:
                ecef_x_list.append(float(col))
            except:
                try:
                    ecef_x_list.append(float(col[:-1]))
                except:
                    continue
    ecef_y_list = []
    ecef_y = file[48:59]
    ecef_z_list = []
    ecef_z = file[59:70]
    for row in ecef_y:
        cols = row.split()
        for col in cols:
            try:
                ecef_y_list.append(float(col))
            except:
                try:
                    ecef_y_list.append(float(col[:-1]))
                except:
                    try:
                        ecef_y_list.append(float(col[1:-1]))
                    except:
                        continue
    for row in ecef_z:
        cols = row.split()
        for col in cols:
            try:
                ecef_z_list.append(float(col))
            except:
                try:
                    ecef_z_list.append(float(col[:-1]))
                except:
                    continue
start_time1 = start_time
end_time1 = end_time
x1 = ecef_x_list
y1 = ecef_y_list
z1 = ecef_z_list

with open('LC08_L1TP_017040_20230314_20230321_02_T1_ANG.txt') as infile:
    file = infile.readlines()
    num_points = file[25].strip()[-2:]
    num_points = int(num_points)
    start_time = file[24][-13:]
    start_time = float(start_time)
    end_time = start_time + num_points
    ecef_x = file[37:48]
    ecef_x_list = []
    for row in ecef_x:
        cols = row.split()
        for col in cols:
            try:
                ecef_x_list.append(float(col))
            except:
                try:
                    ecef_x_list.append(float(col[:-1]))
                except:
                    continue
    ecef_y_list = []
    ecef_y = file[48:59]
    ecef_z_list = []
    ecef_z = file[59:70]
    for row in ecef_y:
        cols = row.split()
        for col in cols:
            try:
                ecef_y_list.append(float(col))
            except:
                try:
                    ecef_y_list.append(float(col[:-1]))
                except:
                    try:
                        ecef_y_list.append(float(col[1:-1]))
                    except:
                        continue
    for row in ecef_z:
        cols = row.split()
        for col in cols:
            try:
                ecef_z_list.append(float(col))
            except:
                try:
                    ecef_z_list.append(float(col[:-1]))
                except:
                    continue
st2 = start_time
et2 = end_time
x2 = ecef_x_list
y2 = ecef_y_list
z2 = ecef_z_list     
     
with open('LC08_L1TP_017041_20230314_20230321_02_T1_ANG.txt') as infile:
    file = infile.readlines()
    num_points = file[25].strip()[-2:]
    num_points = int(num_points)
    start_time = file[24][-13:]
    start_time = float(start_time)
    end_time = start_time + num_points
    ecef_x = file[37:48]
    ecef_x_list = []
    for row in ecef_x:
        cols = row.split()
        for col in cols:
            try:
                ecef_x_list.append(float(col))
            except:
                try:
                    ecef_x_list.append(float(col[:-1]))
                except:
                    continue
    ecef_y_list = []
    ecef_y = file[48:59]
    ecef_z_list = []
    ecef_z = file[59:70]
    for row in ecef_y:
        cols = row.split()
        for col in cols:
            try:
                ecef_y_list.append(float(col))
            except:
                try:
                    ecef_y_list.append(float(col[:-1]))
                except:
                    try:
                        ecef_y_list.append(float(col[1:-1]))
                    except:
                        continue
    for row in ecef_z:
        cols = row.split()
        for col in cols:
            try:
                ecef_z_list.append(float(col))
            except:
                try:
                    ecef_z_list.append(float(col[:-1]))
                except:
                    continue
st3 = start_time
et3 = end_time
x3 = ecef_x_list
y3 = ecef_y_list
z3 = ecef_z_list  

dif1 = int(st2 - start_time1)
xout = x1[:dif1]
yout = y1[:dif1]
zout = z1[:dif1]
dif2 = int(st3 - st2)
xout += x2[:dif2]
yout += y2[:dif2]
zout += z2[:dif2]
xout += x3
yout += y3
zout += z3
t = np.arange(0,102)
out = np.vstack((t,xout,yout,zout)).T

with open('l8_ecef.csv','w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(out)
    
    
    
    
    
    
    
    
    