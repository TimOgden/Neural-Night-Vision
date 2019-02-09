import cv2
import read_files
import shrink_img
import time
from shutil import copyfile

def read_img_file_names(file):
    names = []
    with open(file) as fo:
        for line in fo:
            entry = line.split(' ')
            entry[1] = entry[1][:-1]
            names.append(entry)
    return names

def copy_img(path, data_type, img_type, extension):
    copyfile(path, '../{}/{}/{}'.format(data_type, img_type, extension))
    
def get_greyscale_shrunk_images(names, xres, yres):
    i = 0
    for entry in names:
        for index, file_name in enumerate(entry):
            try:
                img = cv2.imread(file_name, cv2.IMREAD_COLOR)
                img = shrink_img.shrink(img, xres, yres)
                grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite('../images/{}_{}_{}_{}.jpg'.format(i, xres, yres, 'short' if index == 0 else 'long') , grey_img)
            except:
                pass
        i += 1
    
if __name__ == '__main__':
    files = ['../train.txt', '../test.txt', '../val.txt']
    start = time.time()

    end = time.time()
    print('It took ' + str(end - start) + 'seconds')
