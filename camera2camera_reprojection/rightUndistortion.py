import cv2
import os
import numpy as np


# intrisic matrix and distortion coefficient 

camera_matrix = np.array([[6.851146697750740e+02, 0, 6.404138885218853e+02], [0, 6.857579284730023e+02, 3.490960148897398e+02], [0, 0, 1]], dtype=float)
dist_coeffs = np.array([0.137454916221718, -0.100839249073482, 0, 0])  

# dir path

input_folder = '.\\right_mono'  # source dir
output_folder = '.\\right_mono_ud'  # output dir

# if dir dose not exist then mkdir
if not os.path.exists(output_folder):
	os.makedirs(output_folder)

# get all file in input dir
files = os.listdir(input_folder)

for file in files:
	file_path = os.path.join(input_folder, file)
	
	# read image
	img = cv2.imread(file_path)
	
	if img is None:
		continue  # continue if read image fail
	
	# get image size
	h, w = img.shape[:2]
	
	# compute new camera matrix
	new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
	# print(f"new_camera_matrix: \n{new_camera_matrix}")
	# undistortion
	undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
	
	# get roi image
	# x, y, w, h = roi
	# undistorted_img = undistorted_img[y:y+h, x:x+w]
	
	# save undistortion image 
	output_path = os.path.join(output_folder, file)
	cv2.imwrite(output_path, undistorted_img)
	

print("done", output_folder)
