import copy
import math
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import newton_raphson
import read_pcd
import select_lidar_roi
lidar = []
key = None

def pointsCloud_to_scan(points):
	ret = np.zeros((points.shape[0], 2))
	# print(ret.shape)
	ret[:, 0] = np.linalg.norm(points, axis=-1)
	ret[:, 1] = np.arccos(points[:, 0] / ret[:, 0]) - np.pi/2
	
	return ret

def compute_R_imu2lidar(ypr):
	r = Rotation.from_euler("yxz", ypr)
	r = r.as_matrix()
	return r

def show_points(point, roi=(0, 0, 0, 0), roi_color=(0, 0, 255)):
		points = point*100
		points[:, 2] = -1*points[:, 2]
		points = np.stack((point[:, 0], -1*point[:, 2]), axis=-1)*100

		canvas_col_min = np.min(points[:, 0])
		canvas_row_min = np.min(points[:, 1])
		row = int(np.max(points[:, 1]) - canvas_row_min) + 60
		col = int(np.max(points[:, 0]) - canvas_col_min) + 60
		canvas = np.zeros((row, col, 3), dtype=np.uint8)
		# print(row, col, canvas_col_min, canvas_row_min, np.max(points[:, 0]), np.max(points[:, 1]), row, col)

		for i in range(points.shape[0]):
			x = int(points[i, 0] - canvas_col_min) + 30
			y = int(points[i, 1] - canvas_row_min) + 30
			if point[i, 0] < roi[0] or point[i, 0] > roi[1] or point[i, 2] < roi[2] or point[i, 2] > roi[3]:
				cv2.circle(canvas, (x, y), 1, (255, 0, 0))
			else:
				cv2.circle(canvas, (x, y), 1, roi_color)
		# cv2.destroyAllWindows()
		return canvas

ind = 1
data_index =str(ind).zfill(4)
lidar1_point_cloud_path = ".\\data\\PointClouds2\\" + data_index + ".pcd"
imu_path = ".\\data\\Imu\\" + data_index + ".npy"

points = read_pcd.read_pcd(lidar1_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
lidarRoi = select_lidar_roi.LidarRoi(points)
lidarRoi.show_figure()

tmp_mask1 = points[:, 0] >= lidarRoi.x_roi[0]
tmp_mask1 &= points[:, 0] <= lidarRoi.x_roi[1]
tmp_mask2 = points[:, 2] >= lidarRoi.z_roi[0]
tmp_mask2 &= points[:, 2] <= lidarRoi.z_roi[1]
tmp_mask = tmp_mask1 & tmp_mask2
lidar1_point = points[tmp_mask]

pitch = newton_raphson.find_pitch(wall_point=lidar1_point)[0]
pitch = pitch*np.pi/180
imu = np.load(imu_path)
pitch_offset = pitch + imu[0]

lidar_in_scan_type = pointsCloud_to_scan(points)
theta_mask = lidar_in_scan_type[:, 1] < np.pi/6
theta_mask &= lidar_in_scan_type[:, 1] > -np.pi/6

roi = np.min(points[theta_mask][:, 0]), np.max(points[theta_mask][:, 0]), np.min(points[theta_mask][:, 2]), np.max(points[theta_mask][:, 2])
lidarRoi.show_points(points, roi)


# compensate = imu[0]-pitch_offset
# cv2.namedWindow("points", 0)
img_list = []
for indx in range(1, 223):
	# if key == 27:
	# 	break
	ind = indx
	print(ind)
	data_index = str(ind).zfill(4)
	lidar1_point_cloud_path = ".\\data\\PointClouds2\\" + data_index + ".pcd"
	imu_path = ".\\data\\Imu\\" + data_index + ".npy"
	
	imu = np.load(imu_path)
	points = read_pcd.read_pcd(lidar1_point_cloud_path)
	points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
	pitch = imu[0] - pitch_offset
	ypr = np.array([-1*pitch, -1*imu[1], imu[2]])

	points = np.dot(points, compute_R_imu2lidar(ypr=ypr))
	lidar_in_scan_type = pointsCloud_to_scan(points)
	theta_mask = lidar_in_scan_type[:, 1] < np.pi/12
	theta_mask &= lidar_in_scan_type[:, 1] > -np.pi/12

	# lidarRoi = select_lidar_roi.LidarRoi(points)
	roi = np.min(points[theta_mask][:, 0]), np.max(points[theta_mask][:, 0]), np.min(points[theta_mask][:, 2]), np.max(points[theta_mask][:, 2])
	canvas = show_points(points, roi)
	canvas = cv2.resize(canvas, (300, 300))
	# canvas.resize((500, 500, 3))
	img_list.append(canvas)
	cv2.imshow("test", canvas)
	cv2.waitKey(100)
	# cv2.imwrite(".\\test_img\\" + data_index + ".png", canvas)
cv2.destroyAllWindows()
cv2.namedWindow("points", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("points", 300, 300)
# canvas = np.array(img_list)
# cv2.imshow("points", canvas[0])
# cv2.waitKey(0)
# for i in range(len(img_list)):
# 	data_index = str(i+1).zfill(4)
# 	canvas = cv2.imread(".\\test_img\\" + data_index + ".png")
# 	# canvas = cv2.resize(canvas, (300, 300))
# 	# print(canvas.shape)
# 	# print(type(canvas))
# 	# canvas.resize((100, 100, 3))
# 	cv2.imshow("points", canvas)
# 	# key = cv2.waitKey(30)
# 	if cv2.waitKey(30) & 0xFF == ord('q'):
# 		break
# 	# lidarRoi = select_lidar_roi.LidarRoi(points)
# 	# lidarRoi.show_figure()

# 	tmp_mask1 = points[:, 0] >= lidarRoi.x_roi[0]
# 	tmp_mask1 &= points[:, 0] <= lidarRoi.x_roi[1]
# 	tmp_mask2 = points[:, 2] >= lidarRoi.z_roi[0]
# 	tmp_mask2 &= points[:, 2] <= lidarRoi.z_roi[1]
# 	tmp_mask = tmp_mask1 & tmp_mask2
# 	lidar1_point = points[tmp_mask]

# 	lidar1_image = lidarRoi.canvas[lidarRoi.start_y:lidarRoi.end_y, lidarRoi.start_x:lidarRoi.end_x]

# 	# print(lidar1_point)
# 	cv2.namedWindow("lidar1", 0)
# 	cv2.imshow("lidar1", lidar1_image)

# 	key = cv2.waitKey(0)

	# lidar1_sort_id = np.argsort(lidar1_point[:, 0])
	# lidar1_sort = lidar1_point[lidar1_sort_id]
	# lidar.append(lidar1_sort)

	# # lidar1_point = lidar1_point
	# cv2.destroyAllWindows()
# # lidar = np.array(lidar)
# print(len(lidar))
# # print(lidar.shape)
# ans = input("collect lidar points? ")
# if ans == 'y':
# 	directory_name = input("Directory Name: ")
# 	base_path = os.path.join("box", directory_name)
	
# 	if not os.path.exists(base_path):
# 		os.makedirs(base_path)
# 	for i in range(len(lidar)):
# 		ind = str(i).zfill(4)

# 		lidar_points_path = os.path.join(base_path, "box" + ind + ".npy")
# 		data_saved = np.array(lidar[i])
# 		np.save(lidar_points_path, data_saved)
# 	print("done!")