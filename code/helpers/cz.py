from PIL import Image
import os
from tqdm import tqdm

PATH = './cesims2'
OUT_PATH = './interest_ds/test'


folders = os.listdir(PATH)
rois = ['10S','10T','11R','11S','12R','16T','17R','17T',
        '18S','32S','32T','33S','33T','52S','53S','54S',
        '54T']

for folder in tqdm(folders):
    files = os.listdir(os.path.join(PATH,folder))
    if folder in rois:
        for file in files:
            if file.endswith('.png'):
                im = Image.open(os.path.join(PATH,folder,file))
                small_im = im.resize((round(im.width/4.05),round(im.height/4.05)))
                if not os.path.exists(os.path.join(OUT_PATH,folder)):
                    os.mkdir(os.path.join(OUT_PATH,folder))
                    print('made dir', folder)
                small_im.save(os.path.join(OUT_PATH,folder,file))
    else:
        for file in files:
            if file.endswith('.png'):
                im = Image.open(os.path.join(PATH,folder,file))
                small_im = im.resize((round(im.width/4.05),round(im.height/4.05)))
                small_im.save(os.path.join(OUT_PATH,'elsweyr',file))