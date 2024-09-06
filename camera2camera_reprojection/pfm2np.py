import numpy as np
import os
def read_pfm(file):
    file = open(file, 'rb')

    # Read the header to determine if it's a color or grayscale image
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    # Read dimensions
    dimensions = file.readline().decode('utf-8').rstrip()
    width, height = map(int, dimensions.split())

    # Read the scale factor (little-endian or big-endian)
    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:  # big-endian
        endian = '>'

    # Read the data
    data = np.fromfile(file, endian + 'f')
    data = np.reshape(data, (height, width, 3) if color else (height, width)) 

    return data, scale

if __name__ == '__main__':

    input_folder = '.\\test_depth'  # pfm file folder 
    output_folder = '.\\test_depth'    # folder that npy file saved

    # if folder not exist than create it.  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get all pfm file name
    files = os.listdir(input_folder)
    pfm_files = [file for file in files if file.endswith('.pfm')]

    # travel all pfm file and convert to npy file
    for pfm_file in pfm_files:
        input_path = os.path.join(input_folder, pfm_file)
        
        # read pfm file and return npy file 
        array, scale = read_pfm(input_path)
        
        # constructure the npy file name
        npy_file = pfm_file.replace('.pfm', '.npy')
        output_path = os.path.join(output_folder, npy_file)
        
        # save numpy data to npy file
        np.save(output_path, array)
        # remove the pfm file
        os.remove(input_path)