# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:28:52 2023

@author: kmccl
"""

from skyfield.api import load, wgs84, EarthSatellite
import numpy as np
import json


def propagate_tle(tle, start_offset, seconds, fps):
    ts = load.timescale()
    if len(tle) == 3:
        name = tle[0]
        line_1 = tle[1]
        line_2 = tle[2]
        satellite = EarthSatellite(line_1, line_2, name=name, ts=ts)
    elif len(tle) == 2:
        line_1 = tle[0]
        line_2  = tle[1]
        satellite = EarthSatellite(line_1, line_2, name=None, ts=ts)
    else:
        print('not a valid tle format')
        return
    
    t = satellite.epoch + start_offset/86400.0
    positions_llh = []
    epochs = np.linspace(0, seconds, 
                         round(fps*seconds))
    for epoch in epochs:
        t_new = t + epoch/86400.0
        pos = satellite.at(t_new)
        llh = wgs84.geographic_position_of(pos)
        positions_llh.append([llh.longitude.degrees, llh.latitude.degrees,
                             llh.elevation.m])
    return satellite.name, np.array(positions_llh)

tle = ['ISS',
           '1 25544U 98067A   23087.87270052  .00029877  00000-0  53474-3 0  9990',
           '2 25544  51.6420   6.4657 0005772 142.1937 328.5238 15.49707242389334']

start_offset = 201000
seconds = 201110 - start_offset
fps = 1
name, llh_pos = propagate_tle(tle, start_offset, seconds, fps)
lons = llh_pos[:,0]
lats = llh_pos[:,1]
alts = llh_pos[:,2]
lats_lt = lats < 32
lats_gt = lats > 24
lons_lt = lons < -108
lons_gt = lons > -114
lons_in = lons_lt*lons_gt
lats_in = lats_lt*lats_gt
lonlats_in = lons_in*lats_in
print(lonlats_in.sum())
llh_pos = llh_pos[np.where(lonlats_in)]

poses = []
heading = np.random.rand() * 360 - 180
pitch = np.random.rand() * 180 - 180
roll = np.random.rand() * 360 - 180

rot_pitch = np.random.rand() * 10 - 5
rot_roll = np.random.rand() * 10 - 5
for llh in llh_pos:
    lon, lat, alt = llh
    pose = [lon, lat, alt, heading, pitch, roll]
    poses.append(pose)
    pitch += rot_pitch * 1/fps
    roll += rot_roll * 1/fps
    
out = json.dumps(poses)
print(out)