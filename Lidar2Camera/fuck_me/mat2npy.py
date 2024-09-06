import os
import numpy as np
import scipy.io 

import numpy as np

class Mat2NPY():
	def __init__(self):
		pass
	def convertDisp(self, data_name="disp_u32", input_folder=None, output_folder=None):
		if input_folder is None:
			input_folder = ".\\data\\Disp"
		if output_folder is None:
			output_folder = ".\\data\\Disp"
		# if not exist then mkdir
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		# get all .mat file
		files = os.listdir(input_folder)
		mat_files = [file for file in files if file.endswith('.mat')]

		# traval all .mat file and convert to .npy file 
		for mat_file in mat_files:
			input_path = os.path.join(input_folder, mat_file)
			# transform mat to np file
			mat_data = scipy.io.loadmat(input_path)
			array = mat_data[data_name]
			array = array.reshape(-1).astype(np.uint8)
			# Reshape the array into a 2D array with 4 columns
			reshaped_array = array.astype(np.uint32).reshape(-1, 4)

			# Combine the 4 uint8 values into a single uint32 value
			# little-endian format
			uint32_data = reshaped_array[:, 0] | (reshaped_array[:, 1] << 8) | (reshaped_array[:, 2] << 16) | (reshaped_array[:, 3] << 24)

			new_array = np.frombuffer(uint32_data.tobytes(), dtype=np.uint32).view(np.float32)
			array = new_array
			# reconstruct file name
			npy_file = mat_file.replace('.mat', '.npy')
			output_path = os.path.join(output_folder, npy_file)
			
			# store npy file
			np.save(output_path, array)
			os.remove(input_path)

	def convertDepth(self, data_name="depth_u32", input_folder=None, output_folder=None):
		if input_folder is None:
			input_folder = ".\\data\\Depth"
		if output_folder is None:
			output_folder = ".\\data\\Depth"
		# if not exist then mkdir
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		# get all .mat file
		files = os.listdir(input_folder)
		mat_files = [file for file in files if file.endswith('.mat')]

		# traval all .mat file and convert to .npy file 
		for mat_file in mat_files:
			input_path = os.path.join(input_folder, mat_file)
			# transform mat to np file
			mat_data = scipy.io.loadmat(input_path)
			array = mat_data[data_name]
			array = array.reshape(-1).astype(np.uint8)
			# Reshape the array into a 2D array with 4 columns
			reshaped_array = array.astype(np.uint32).reshape(-1, 4)

			# Combine the 4 uint8 values into a single uint32 value
			# little-endian format
			uint32_data = reshaped_array[:, 0] | (reshaped_array[:, 1] << 8) | (reshaped_array[:, 2] << 16) | (reshaped_array[:, 3] << 24)

			new_array = np.frombuffer(uint32_data.tobytes(), dtype=np.uint32).view(np.float32)
			array = new_array
			# reconstruct file name
			npy_file = mat_file.replace('.mat', '.npy')
			output_path = os.path.join(output_folder, npy_file)
			
			# store npy file
			np.save(output_path, array)
			os.remove(input_path)

	def converImu(self, data_name="ypr", input_folder=None, output_folder=None):
		input_folder = '.\\data\\Imu'
		output_folder = '.\\data\\Imu'
		
		# if not exist then mkdir
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)

		# get all .mat file
		files = os.listdir(input_folder)
		mat_files = [file for file in files if file.endswith('.mat')]

		# traval all .mat file and convert to .npy file 
		for mat_file in mat_files:
			input_path = os.path.join(input_folder, mat_file)
			
			# transform mat to np file
			mat_data = scipy.io.loadmat(input_path)

			array = mat_data[data_name]
			array = array.reshape(-1).astype(np.float32)
			
			# reconstruct file name
			npy_file = mat_file.replace('.mat', '.npy')
			output_path = os.path.join(output_folder, npy_file)
			
			# store npy file
			np.save(output_path, array)
			os.remove(input_path)


if __name__ == "__main__":
	mat2npy = Mat2NPY()
	# mat2npy.convertDisp(input_folder=".\\test\\Disp", output_folder=".\\test\\Disp")
	# mat2npy.convertDepth(input_folder=".\\test\\Depth", output_folder=".\\test\\Depth")
	mat2npy.converImu()
