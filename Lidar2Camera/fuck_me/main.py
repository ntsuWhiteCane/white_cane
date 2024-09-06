import copy
import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import newton_raphson
import read_pcd
import select_lidar_roi
import newton_raphson

def compute_R_imu2lidar(ypr):
	r = Rotation.from_euler("yxz", ypr)
	r = r.as_matrix()
	return r

ind = 80
data_index = str(ind).zfill(4)
lidar1_point_cloud_path = ".\\data\\PointClouds1\\" + data_index + ".pcd"
lidar2_point_cloud_path = ".\\data\\PointClouds2\\" + data_index + ".pcd"
image_path = ".\\data\\Images\\" + data_index + ".png"
imu_path = ".\\data\\Imu\\" + data_index + ".npy"

imu = np.load(imu_path)

points = read_pcd.read_pcd(lidar1_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
points = points[points[:, 2] < 3]
points_sort_id = np.argsort(points[:, 0])
points = points[points_sort_id]

lidarRoi = select_lidar_roi.LidarRoi(points)
lidarRoi.show_figure()


roi = (np.min((lidarRoi.x_roi[0], lidarRoi.x_roi[1])), np.max((lidarRoi.x_roi[0], lidarRoi.x_roi[1])), np.min((lidarRoi.z_roi[0], lidarRoi.z_roi[1])), np.max((lidarRoi.z_roi[0], lidarRoi.z_roi[1])))

tmp_mask1 = points[:, 0] >= lidarRoi.x_roi[0] 
tmp_mask1 &= points[:, 0] <= lidarRoi.x_roi[1]
tmp_mask2 = points[:, 2] >= lidarRoi.z_roi[0] 
tmp_mask2 &= points[:, 2] <= lidarRoi.z_roi[1]
wall_mask = tmp_mask1 & tmp_mask2

lidar1_wall = points[wall_mask]
# print(p.shape)
# lidarRoi = select_lidar_roi.LidarRoi(p)
# lidarRoi.show_figure()
# lidarRoi.show_points(p)

pitch = newton_raphson.find_pitch(lidar1_wall)
pitch = pitch[0]*np.pi/180
# print(pitch)
# rotation_matrix = np.array([ [np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
# print()
# points = np.dot(points, rotation_matrix.T)

# lidarRoi.show_points(p)

print(imu)
# imu system is z y x, but in lidar system is -y -x z
ypr = np.array([-1*pitch, -1*imu[1], imu[2]])

lidar1_wall = np.dot(points, compute_R_imu2lidar(ypr=ypr))
print(lidar1_wall)
lidarRoi.show_points(lidar1_wall)
cv2.destroyAllWindows()

points = read_pcd.read_pcd(lidar2_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
points = points[points[:, 2] < 3]
points_sort_id = np.argsort(points[:, 0])
points = points[points_sort_id]

lidarRoi = select_lidar_roi.LidarRoi(points)
lidarRoi.show_figure()


roi = (np.min((lidarRoi.x_roi[0], lidarRoi.x_roi[1])), np.max((lidarRoi.x_roi[0], lidarRoi.x_roi[1])), np.min((lidarRoi.z_roi[0], lidarRoi.z_roi[1])), np.max((lidarRoi.z_roi[0], lidarRoi.z_roi[1])))

tmp_mask1 = points[:, 0] >= lidarRoi.x_roi[0] 
tmp_mask1 &= points[:, 0] <= lidarRoi.x_roi[1]
tmp_mask2 = points[:, 2] >= lidarRoi.z_roi[0] 
tmp_mask2 &= points[:, 2] <= lidarRoi.z_roi[1]
wall_mask = tmp_mask1 & tmp_mask2

lidar2_wall = points[wall_mask]

lidar2_wall = np.dot(lidar2_wall, compute_R_imu2lidar(ypr=ypr))

lidarRoi.show_points(lidar2_wall)
cv2.destroyAllWindows()
newton_raphson.wall_fit(lidar1_wall, lidar2_wall, [0, 0, 0, 0.2])
