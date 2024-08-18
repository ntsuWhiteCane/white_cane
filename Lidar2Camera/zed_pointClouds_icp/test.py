import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import open3d as o3d

import select_lidar_roi
import select_camera_roi
# from mpl_toolkits.mplot3d import Axes3D

ind = 9
data_index = str(ind).zfill(4)
zed_image_path = "Images\\" + data_index + ".png"
lidar_point_cloud_path = "PointClouds\\" + data_index + ".pcd"
zed_point_cloud_path = "ZED_Point_Clouds\\" + data_index + ".pcd"
depth_path = "Depth\\" + data_index + ".npy"

zed_K = np.array([[3.506789914489589e+02, 0, 3.127677577917245e+02],
                  [0, 3.510410495486862e+02, 1.875433445883912e+02],
                  [0, 0, 1]])

def read_pcd(file_path):
	with open(file_path, 'r') as f:
		data = f.readlines()

	# Find start of the point data
	for i, line in enumerate(data):
		if line.startswith('DATA'):
			data_start = i + 1
			break

	# Read point data
	points = []
	for line in data[data_start:]:
		point = line.split()
		if float(point[0]) < 0 or math.isnan(float(point[0])):
			continue
		points.append([float(point[0]), float(point[1]), float(point[2])])  # Assuming X Y Z fields

	points = np.array(points)

	return points

points = read_pcd(lidar_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)

lidarRoi = select_lidar_roi.LidarRoi(points)
lidarRoi.show_figure()

tmp_mask1 = points[:, 0] >= lidarRoi.x_roi[0]
tmp_mask1 &= points[:, 0] <= lidarRoi.x_roi[1]
tmp_mask2 = points[:, 2] >= lidarRoi.z_roi[0]
tmp_mask2 &= points[:, 2] <= lidarRoi.z_roi[1]
tmp_mask = tmp_mask1 & tmp_mask2

lidar_point = points[tmp_mask]
# print(lidar_point)

lidar_pcd = o3d.geometry.PointCloud()
lidar_pcd.points = o3d.utility.Vector3dVector(lidar_point)

depth = np.load(depth_path)
depth = depth * 1000
print(depth[180, 320])
img = cv2.imread(zed_image_path, -1)

points = read_pcd(zed_point_cloud_path) * 1000
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)

cameraRoi = select_camera_roi.CameraRoi(img, points, zed_K)
cameraRoi.show_figure()
points = cameraRoi.computePoint()

# print(points.shape)

zed_pcd = o3d.geometry.PointCloud()
zed_pcd.points = o3d.utility.Vector3dVector(points)

trans_init = np.asarray([[1.0, 0.0, 0.0, 0.075],
						 [0.0, 1.0, 0.0, 0.295],
						 [0.0, 0.0, 1.0, 0.4],
						 [0.0, 0.0, 0.0, 1.0]])
reg_p2p = o3d.pipelines.registration.registration_icp(lidar_pcd, zed_pcd, 0.2, trans_init, 
														o3d.pipelines.registration.TransformationEstimationPointToPoint(),
														o3d.pipelines.registration.ICPConvergenceCriteria(1))
print(reg_p2p)
print(reg_p2p.transformation, "\n")

print("zed pcd:", zed_pcd)
print("lidar pcd:", lidar_pcd)
print()
# pcd_filtered = pcd.uniform_down_sample(every_k_points=3)

zed_filter = zed_pcd.uniform_down_sample(every_k_points=10)
o3d.visualization.draw_geometries(geometry_list = [zed_filter],  window_name="point_cloud")

np.save("transformation.npy", reg_p2p.transformation)

