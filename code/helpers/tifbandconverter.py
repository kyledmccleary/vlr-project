import rasterio
import cv2
import os
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
# import sys

# debug
# sys.argv = ['tifbandconverter.py', '-p', 'RGBNIR', '-m', 'rgbn2gs', '-o', 'rgbn2gs']

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path_to_folder', required=True, type=str)
parser.add_argument('-m', '--mode', choices=['rgb2gs', 'rgbn2gs', 'rgbn2rgb', 'rgbn2nrg', 'rgbn23c'], required=True)
parser.add_argument('-o', '--out_path', required=True, type=str)
args = parser.parse_args()
path = args.path_to_folder
mode = args.mode


def process_image(filepath):
    if filepath.endswith('tif'):
        with rasterio.open(os.path.join(path, filepath)) as src:
            im = src.read()
            im = im.transpose((1, 2, 0))
            if mode == 'rgb2gs':
                im = im[:, :, :3]
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                out = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            elif mode == 'rgbn2gs':
                im = np.float32(im)
                out = im[:, :, 0] * 0.25 + im[:, :, 1] * 0.25 + im[:, :, 2] * 0.25 + im[:, :, 3] * 0.25
            elif mode == 'rgbn2rgb':
                im = im[:, :, :3]
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                out = im
            elif mode == 'rgbn2nrg':
                out = np.stack((im[:, :, 3], im[:, :, 0], im[:, :, 1]), axis=-1)
                out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            elif mode == 'rgbn23c':
                rn = np.float32(im)
                rn = rn[:, :, 0]*0.5 + rn[:, :, 3]*0.5
                out = np.stack((im[:, :, 2], im[:, :, 1], rn), axis=-1)
        out_path = os.path.join(args.out_path, filepath[:-4] + '.jpg')
        if not os.path.exists(args.out_path):
            os.mkdir(args.out_path)
            print('created directory:', args.out_path)
        cv2.imwrite(out_path, out)
        print(out_path, 'done')


if __name__ == '__main__':
    files = os.listdir(path)
    with Pool(cpu_count()) as p:
        p.map(process_image, files)
