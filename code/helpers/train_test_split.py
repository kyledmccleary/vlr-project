from sklearn.model_selection import train_test_split
import os
import shutil

im_path = 'datasets/it/images'
lab_path = 'datasets/it/algo_labels'

seed = 0

files = os.listdir(im_path)
images = []
for file in files:
    if file.endswith('.png'):
        images.append(file)

files = os.listdir(lab_path)
labels = []
for file in files:
    if file.endswith('.txt'):
        labels.append(file)

X_train, X_val, y_train, y_val = train_test_split(images,labels,test_size = 0.2,shuffle=True,random_state = seed)

val_im_path = os.path.join(im_path,'val')
train_im_path = os.path.join(im_path,'train')
val_lab_path = os.path.join(lab_path,'val')
train_lab_path = os.path.join(lab_path,'train')

if not os.path.exists(val_im_path):
    os.makedirs(val_im_path)
if not os.path.exists(train_im_path):
    os.makedirs(train_im_path)
if not os.path.exists(val_lab_path):
    os.makedirs(val_lab_path)
if not os.path.exists(train_lab_path):
    os.makedirs(train_lab_path)

for X in X_train:
    source = os.path.join(im_path,X)
    dest = os.path.join(train_im_path,X)
    dest = shutil.move(source,dest)
for y in y_train:
    source = os.path.join(lab_path,y)
    dest = os.path.join(train_lab_path,y)
    dest = shutil.move(source,dest)
for X in X_val:
    source = os.path.join(im_path,X)
    dest = os.path.join(val_im_path,X)
    dest = shutil.move(source,dest)
for y in y_val:
    source = os.path.join(lab_path,y)
    dest = os.path.join(val_lab_path,y)
    dest = shutil.move(source,dest)