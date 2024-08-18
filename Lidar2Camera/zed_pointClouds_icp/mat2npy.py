import os
import numpy as np
import scipy.io 

if __name__ == '__main__':

    input_folder = '.\\Depth'
    output_folder = '.\\Depth'
    
    # if not exist then mkdir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # get all .mat file
    files = os.listdir(input_folder)
    pfm_files = [file for file in files if file.endswith('.mat')]

    # traval all .mat file and convert to .npy file 
    for mat_file in pfm_files:
        input_path = os.path.join(input_folder, mat_file)
        
        # transform mat to np file
        mat_data = scipy.io.loadmat(input_path)
        array = mat_data['depth_u32']
        
        # reconstruct file name
        npy_file = mat_file.replace('.mat', '.npy')
        output_path = os.path.join(output_folder, npy_file)
        
        # store npy file
        np.save(output_path, array)
        os.remove(input_path)