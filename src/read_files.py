def get_jpg_names_from_file(file):
    updated_file_names = []
    with open(file) as fo:
        for line in fo:
            parsed_line = line.split(' ')
            for i in range(2):
                file_name = parsed_line[i].split('.')
                file_name.pop()
                file_name.append('jpg')
                parsed_line[i] = '.'.join(file_name)
            updated_file_names.append(parsed_line)
    return updated_file_names

def change_abs_dir(l, new_dir):
    updated_file_names = []
    for entry in l:
        new_entry = []
        for file_name in entry:
            changed_fn = file_name.split('/')
            changed_fn = [new_dir] + changed_fn[2:]
            changed_fn = '/'.join(changed_fn)
            new_entry.append(changed_fn)
        updated_file_names.append(new_entry)
    return updated_file_names
    
    
def get_images_from_list(l):
    images = []
    for entry in l:
        images.append(entry[:2])
    return images

def get_images_from_file(file):
    return change_abs_dir(get_images_from_list(get_jpg_names_from_file(file)), '..')
    
def print_list(l):
    print('[')
    for i in range(len(l)):
        print(str(l[i]) + (',' if i != len(l) - 1 else ''))
    print(']')

def create_updated_list(read_file, write_file):
    img_names = change_abs_dir(get_images_from_list(get_jpg_names_from_file(read_file)), '../Sony')
    f = open(write_file, 'w')
    for entry in img_names:
        f.write(entry[0] + ' ' + entry[1] + '\n')
    
if __name__ == '__main__':
    inputs = ['../Sony_train_list.txt', '../Sony_test_list.txt', '../Sony_val_list.txt']
    outputs = ['../train.txt', '../test.txt', '../val.txt']
    for i in range(len(inputs)):
        create_updated_list(inputs[i], outputs[i])
    pass
    '''
    data = get_jpg_names_from_file('../Sony_train_list.txt')
    images_name = get_images_from_list(data)
    images_name = change_abs_dir(images_name, '..')
    f = open('img_names.txt', 'w')
    for entry in images_name:
        f.write(entry[0] + ' ' + entry[1] + '\n')
    '''


