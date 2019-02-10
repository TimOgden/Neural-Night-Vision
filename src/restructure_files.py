import os
import file_loader
from os.path import exists

def create_files():
    names = ['train', 'val', 'test']
    types = ['short', 'long']
    for name in names:
        os.makedirs('../{}'.format(name))
        for type in types:
            os.makedirs('../{}/{}'.format(name, type))

def reorganize_files():
    txt_files = ['../train.txt', '../val.txt', '../test.txt']
    data_type = {0: 'train', 1: 'test', 2: 'val'}
    for i, txt_file in enumerate(txt_files):
        names = file_loader.read_img_file_names(txt_file)
        i = 0
        for entry in names:
            for index, file_name in enumerate(entry):
                file_loader.copy_img(file_name, data_type[i], 'short' if not index else 'long', i)
            i += 1
        
def remove_nonexistent_files():
    txt_files = ['../train.txt', '../val.txt', '../test.txt']
    new_txt_files = ['../new_train.txt', '../new_val.txt', '../new_test.txt']
    for i, txt_file in enumerate(txt_files):
        with open(txt_file) as fo:
            with open(new_txt_files[i], 'w') as new_fo:
                for line in fo:
                    # get rid of new line in entry
                    entry = line.split(' ')
                    entry[1] = entry[1][:-1]
                    if exists(entry[0]) and exists(entry[1]):
                        new_fo.write(entry[0] + ' ' + entry[1] + '\n')

if __name__ == '__main__':
    remove_nonexistent_files()
