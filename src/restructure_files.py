import os
import file_loader

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
        
if __name__ == '__main__':
    try:
        create_files()
    except:
        pass
    reorganize_files()
