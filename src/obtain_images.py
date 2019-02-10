from file_loader import read_img_file_names
import cv2
import numpy as np
from os.path import exists

def obtain_data(filename, amount=-1, transform=lambda x: x):
    #names = n x 2 array
    names = read_img_file_names(filename)

    short_exposures = []
    long_exposures = []
   
    i = 0
    for row in names:
        if amount != -1 and i >= amount:
            break
        short_fn = row[0]
        long_fn = row[1]
        if exists(short_fn) and exists(long_fn):
            short_exposures.append(transform(cv2.imread(short_fn, cv2.IMREAD_COLOR)))
            long_exposures.append(transform(cv2.imread(long_fn, cv2.IMREAD_COLOR)))
        i += 1
    return (np.array(short_exposures), np.array(long_exposures))
            
def shrink_func(xres, yres):
    return lambda x: cv2.resize(x, dsize=(xres, yres), interpolation=cv2.INTER_CUBIC)

def greyscale_func():
    return lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

def shrink_greyscale_func(xres, yres, nchannels):
    if nchannels == 1:
        return lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), dsize=(xres, yres), interpolation=cv2.INTER_CUBIC)
    else:
        return shrink_func(xres, yres)

if __name__ == '__main__':
    obtain_data('../train.txt', amount=32)
