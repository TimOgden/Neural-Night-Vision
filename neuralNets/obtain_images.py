from file_loader import read_img_file_names
import cv2
from os.path import exists

def obtain_data(filename, amount=-1):
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
            short_exposures.append(cv2.imread(short_fn, cv2.IMREAD_COLOR))
            long_exposures.append(cv2.imread(long_fn, cv2.IMREAD_COLOR))
        i += 1
    return (short_exposures, long_exposures)
            
if __name__ == '__main__':
    obtain_data('../train.txt')