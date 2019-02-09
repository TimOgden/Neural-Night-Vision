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

def print_list(l):
    for entry in l:
        print(entry)
        
print_list(get_jpg_names_from_file('../Sony_train_list.txt'))

