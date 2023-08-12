from skyfield.api import load, EarthSatellite, wgs84
import numpy as np
import json

ts = load.timescale()


name = 'ISS (ZARYA)'
tle_line1 = '1 25544U 98067A   23190.57783146  .00010473  00000+0  19367-3 0  9999'
tle_line2 = '2 25544  51.6406 217.5536 0000239  93.5169  53.6825 15.49635129405261'

satellite = EarthSatellite(tle_line1, tle_line2, name, ts)
t = satellite.epoch
year = t.utc.year
month = t.utc.month
day = t.utc.day
hour = t.utc.hour
minute = t.utc.minute
second = t.utc.second

latitudes = []
longitudes = []
heights = []

fps = 1
for i in range(0, 300, fps):
    t = ts.utc(year, month, day, hour, minute, second+i)
    pos = satellite.at(t)
    lat, lon = wgs84.latlon_of(pos)
    height = wgs84.height_of(pos)
    latitudes.append(lat.degrees)
    longitudes.append(lon.degrees)
    heights.append(height.m)
positions = np.vstack((longitudes, latitudes, heights)).T.tolist()
print(positions)
epoch = str(year)+'-'+str(month).zfill(2) + '-' + str(day).zfill(2) \
    + 'T' + str(hour).zfill(2) + ':' + str(minute).zfill(2) + ':' \
    + str(round(second)).zfill(2) + 'Z'

dictionary = {
    "id": "document",
    "name": "simple",
    "version": "1.0",
    "clock": {
        "currentTime": epoch,
        "multiplier": 1,
        "step": "SYSTEM_CLOCK_MULTIPLIER",
    }
}


pos_dict = {
    "interpolationAlgorithm": "LAGRANGE",
    "interpolationDegree": 5,
    "referenceFrame": "INERTIAL",
    "epoch": epoch,
    "cartographicDegrees": positions
}

dict = {
    "id": name,
    "position": pos_dict
}

json_object = json.dumps([dictionary, dict], indent=4)
print(json_object)

with open("sample.czml", 'w') as outfile:
    outfile.write(json_object)