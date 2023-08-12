from PIL import Image
import os
from tqdm import tqdm

PATH = 'F:\landsat_images'
OUT_PATH = './interest_ds/train'


folders = os.listdir(PATH)

for folder in tqdm(folders):
    if os.path.isdir(os.path.join(PATH,folder)):
        files = os.listdir(os.path.join(PATH,folder))
        for file in files:
            im = Image.open(os.path.join(PATH,folder,file))
            small_im = im.resize((round(im.width/4.05),round(im.height/4.05)))
            if not os.path.exists(os.path.join(OUT_PATH,folder)):
                os.mkdir(os.path.join(OUT_PATH,folder))
                print('made dir folder')
            small_im.save(os.path.join(OUT_PATH,folder,file))
