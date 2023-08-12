import argparse
import ee
import requests
import os
import shutil
import numpy as np
from retry import retry
from multiprocessing import Pool, cpu_count


def get_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    return args



def main():
    ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
    args = get_args()

