import copy
import math

import cv2
import numpy as np

import newton_raphson
import read_pcd
import select_lidar_roi
import newton_raphson

ind = 33
data_index = str(ind).zfill(4)
lidar1_point_cloud_path = ".\\data\\PointClouds1\\" + data_index + ".pcd"
lidar2_point_cloud_path = ".\\test_data\\PointClouds2\\" + data_index + ".pcd"
image_path = ".\\test_data\\Images\\" + data_index + ".png"


points = read_pcd.read_pcd(lidar1_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
points = points[points[:, 2] < 3]
points_sort_id = np.argsort(points[:, 0])
points = points[points_sort_id]

lidarRoi = select_lidar_roi.LidarRoi(points)
lidarRoi.show_figure()


roi = (np.min((lidarRoi.x_roi[0], lidarRoi.x_roi[1])), np.max((lidarRoi.x_roi[0], lidarRoi.x_roi[1])), np.min((lidarRoi.z_roi[0], lidarRoi.z_roi[1])), np.max((lidarRoi.z_roi[0], lidarRoi.z_roi[1])))
print(roi)
tmp_mask1 = points[:, 0] >= lidarRoi.x_roi[0] 
tmp_mask1 &= points[:, 0] <= lidarRoi.x_roi[1]
tmp_mask2 = points[:, 2] >= lidarRoi.z_roi[0] 
tmp_mask2 &= points[:, 2] <= lidarRoi.z_roi[1]
box_mask = tmp_mask1 & tmp_mask2

p = points[box_mask]
print(p.shape)
# lidarRoi = select_lidar_roi.LidarRoi(p)
# lidarRoi.show_figure()
lidarRoi.show_points(p)

pitch = newton_raphson.find_pitch(p)
pitch = pitch[0]*np.pi/180
# print(pitch)
rotation_matrix = np.array([ [np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
print()
p = np.dot(p, rotation_matrix.T)

lidarRoi.show_points(p)