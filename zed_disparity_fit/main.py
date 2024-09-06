import os
import numpy as np

input_folder = ".\\data\\Depth\\"
output_folder = ".\\data\\Depth\\"

file_extension = ".mat"

files = os.listdir(input_folder)
count = 0
for i in range(500):
	ind = str(i).zfill(4)
	path_name = ind + file_extension
	new_path_name = str(20+count*10) + file_extension
	if path_name in files:
		os.rename(input_folder + path_name, output_folder + new_path_name)
		print(path_name, new_path_name)
		count += 1
print(files)