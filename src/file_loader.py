import cv2
import read_files
import shrink_img
import time

def read_img_file_names(file):
    names = []
    with open(file) as fo:
        for line in fo:
            entry = line.split(' ')
            entry[1] = entry[1][:-1]
            names.append(entry)
    return names
    
def get_greyscale_shrunk_images(l, xres, yres):
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
    names = read_img_file_names('img_names.txt')
    start = time.time()
    get_greyscale_shrunk_images(names, 64, 64)
    end = time.time()
    print('It took ' + str(end - start) + 'seconds')
