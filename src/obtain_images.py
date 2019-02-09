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
            
if __name__ == '__main__':
    obtain_data('../train.txt', amount=32)
