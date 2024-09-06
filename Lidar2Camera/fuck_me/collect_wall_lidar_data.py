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
for indx in range(1, 211):
	if key == 27:
		break
	ind = indx
	print(ind)
	data_index = str(ind).zfill(4)
	lidar1_point_cloud_path = ".\\tilt_box_data\\PointClouds2\\" + data_index + ".pcd"

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

	lidar1_image = lidarRoi.canvas[lidarRoi.start_y:lidarRoi.end_y, lidarRoi.start_x:lidarRoi.end_x]

	# print(lidar1_point)
	cv2.namedWindow("lidar1", 0)
	cv2.imshow("lidar1", lidar1_image)

	key = cv2.waitKey(0)

	lidar1_sort_id = np.argsort(lidar1_point[:, 0])
	lidar1_sort = lidar1_point[lidar1_sort_id]
	lidar.append(lidar1_sort)

	# lidar1_point = lidar1_point
	cv2.destroyAllWindows()
# lidar = np.array(lidar)
print(len(lidar))
# print(lidar.shape)
ans = input("collect lidar points? ")
if ans == 'y':
	directory_name = input("Directory Name: ")
	base_path = os.path.join("tilt_box", directory_name)
	
	if not os.path.exists(base_path):
		os.makedirs(base_path)
	for i in range(len(lidar)):
		ind = str(i).zfill(4)

		lidar_points_path = os.path.join(base_path, "tilt_box" + ind + ".npy")
		data_saved = np.array(lidar[i])
		np.save(lidar_points_path, data_saved)
	print("done!")