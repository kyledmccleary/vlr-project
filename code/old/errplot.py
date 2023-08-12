import numpy as np
import matplotlib.pyplot as plt
import os

# name = '430x256'
# name = '800x600'
# name = 'dset100'
# name = '430x256_noclip'
# name = 'modis2'
# name = 'modis2_lr1e3'
name = 'modis2_noweights'

path = os.path.join('../data3/results',name)



# tvle = np.load(os.path.join(path,'train_val_loss_err.npy'))
train_loss = np.load(os.path.join(path,'train_loss.npy'))
train_val_error = np.load(os.path.join(path,'train_val_err.npy'))
tad = np.load(os.path.join(path,'train_abs_diffs.npy'))
vad = np.load(os.path.join(path,'val_abs_diffs.npy'))

train_err = train_val_error[:,0]
val_err = train_val_error[:,1]

# lon_range = 18
# lat_range = 24

lon_range = 12.36980146232581
lat_range = 13.429813497586846

# lon_range = 360
# lat_range = 180

###
if len(tad.shape) < 2:
    tad = tad.reshape((tad.shape[0]//8,8))
    vad = vad.reshape((vad.shape[0]//8,8))

new_rows = []
for row in tad:
    lons = row[0:8:2]
    lats = row[1:9:2]
    lons = lons*lon_range
    lats = lats*lat_range
    new_row = [0,0,0,0,0,0,0,0]
    new_row[0:8:2] = lons
    new_row[1:9:2] = lats
    new_row_arr = np.array(new_row)
    new_rows.append(new_row_arr)
new_rows = np.array(new_rows)

new_rows_val = []
for row in vad:
    lons = row[0:8:2]
    lats = row[1:9:2]
    lons = lons*lon_range
    lats = lats*lat_range
    new_row = [0,0,0,0,0,0,0,0]
    new_row[0:8:2] = lons
    new_row[1:9:2] = lats
    new_row_arr = np.array(new_row)
    new_rows_val.append(new_row_arr)
new_rows_val = np.array(new_rows_val)
    

mean_train_lat_lon_err = new_rows.mean(axis=1)
mean_val_lat_lon_err = new_rows_val.mean(axis=1)

fig1, ax1 = plt.subplots()
ax1.plot(train_loss)
ax1.set_title('train loss for ' + name)
fig1.savefig(os.path.join(path,'train_loss.png'),bbox_inches='tight')

fig2, ax2 = plt.subplots()
ax2.plot(train_err)
ax2.plot(val_err)
ax2.set_title('train and val mse for ' + name)
ax2.legend(['train','val'])
ax2.set_ylabel('MSE')
fig2.savefig(os.path.join(path,'train_val_mse.png'),bbox_inches='tight')

fig3, ax3 = plt.subplots()
ax3.plot(mean_train_lat_lon_err)
ax3.plot(mean_val_lat_lon_err)
ax3.text(0.4,0.5, 'min train avg lat lon err:' + str(mean_train_lat_lon_err.min().round(3)),transform = ax3.transAxes)
ax3.text(0.4,0.4, 'min val avg lat lon err:' + str(mean_val_lat_lon_err.min().round(3)),transform = ax3.transAxes)
ax3.set_title('train and val avg lat lon err for ' + name)
ax3.legend(['train','val'])
ax3.set_ylabel('avg degree error')
fig3.savefig(os.path.join(path,'train_val_lat_lon_err.png'),bbox_inches='tight')
