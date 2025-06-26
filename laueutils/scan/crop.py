# To insert a function to remove duplicates OK
import os, shutil
# ----------------------------------------------------------------------


def convertidx_position_to_file(position_index: list, scan_shape: tuple):
    #position index should be an array
    # [[10, 10], [10,11], [10,12], ... ,[11,10], ...]
    file_index = []
    for index in position_index:
        file_index.append(index[0] * scan_shape[0] + index[1])
    return file_index

def copy_and_rename(fileabspath, destination_folder, new_name):
    shutil.copy(fileabspath, destination_folder)
    old_name = os.path.join(destination_folder, fileabspath.split('/')[-1])
    new_name = os.path.join(destination_folder, new_name)
    shutil.move(old_name, new_name)
    

if __name__ == '__main__':
    position_indices = []
    # portion of the scan 10 by 10 at the center of image
    for x in range(30,50):
        for y in range(30,50):
            position_indices.append((x,y))
    
    file_indices = convertidx_position_to_file(position_indices, (81,81))
    
    source_folder      = os.path.join(os.getcwd(), 'datfiles')
    destination_folder = os.path.join(os.getcwd(), 'datfiles_10x10um')
    os.mkdir(destination_folder)
    
    for new_index, curr_index in enumerate(file_indices):
        filename     = f'img_{curr_index:0>4d}.dat'
        fileabspath  = os.path.join(source_folder, filename)
        new_filename = f'img_{new_index:0>4d}.dat'
        copy_and_rename(fileabspath, destination_folder, new_filename)