import math
import os

import cv2
import numpy as np

import newton_raphson
import read_pcd
import select_lidar_roi

threshold = 0.01

ind = 20
data_index = str(ind).zfill(4)
lidar1_point_cloud_path = ".\\lidar2_to_cam\\PointClouds2\\" + data_index + ".pcd"

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

print("lidar1.shape: ", lidar1_point.shape)
cv2.waitKey(0)

lidar1_sort_id = np.argsort(lidar1_point[:, 0])
lidar1_sort = lidar1_point[lidar1_sort_id]

# lidar1_point = lidar1_point
cv2.destroyAllWindows()
lidarRoi = select_lidar_roi.LidarRoi(lidar1_point)
# lidarRoi.find3Points()
lidar1 = lidarRoi.select_3_points()

cv2.namedWindow("3 points", 0)
cv2.imshow("3 points", lidarRoi.canvas)
cv2.waitKey(0) 

cv2.destroyAllWindows()
print(lidar1)

A_lidar = np.array([lidar1[0, 0], lidar1[0, 1], 0])
B_lidar = np.array([lidar1[1, 0], lidar1[1, 1], 0])
D_lidar = np.array([lidar1[2, 0], lidar1[2, 1], 0])
C_lidar = np.array([0, 0, 0])
AB = np.linalg.norm(A_lidar-B_lidar)
BD = np.linalg.norm(B_lidar-D_lidar)
DA = np.linalg.norm(D_lidar-A_lidar)

k1 = np.power(AB, 2)
k2 = np.power(DA, 2)
k3 = np.power(BD, 2)

A_w = np.array([np.sqrt((k1+k2-k3)/2), 0, 0])
B_w = np.array([0, np.sqrt((k1-k2+k3)/2), 0])
D_w = np.array([0, 0, 1*np.sqrt((k2+k3-k1)/2)])
C_w = newton_raphson.find_it(A_point=A_w, B_point=B_w, D_point=D_w, len_a=np.linalg.norm(A_lidar), len_b=np.linalg.norm(B_lidar), len_d=np.linalg.norm(D_lidar))
# # C_lidar = newton_raphson.newton_raphson(A_point=A_w, B_point=B_w, D_point=D_w, len_a=np.linalg.norm(A_lidar), len_b=np.linalg.norm(B_lidar), len_d=np.linalg.norm(D_lidar))
# # print(C_w)
A_lidar = np.array([lidar1[0, 0], 0, lidar1[0, 1]])
B_lidar = np.array([lidar1[1, 0], 0, lidar1[1, 1]])
D_lidar = np.array([lidar1[2, 0], 0, lidar1[2, 1]])
C_lidar = np.array([0, 0, 0])

world_points = np.stack((A_w, B_w, C_w, D_w))
lidar_points = np.stack((A_lidar, B_lidar, C_lidar, D_lidar))

# lidar_points = lidar_points - C_w

world_points_mass_center = (A_w + B_w + C_w + D_w)/4
lidar_points_mass_center = (A_lidar + B_lidar + C_lidar + D_lidar)/4

world_points_new = world_points - world_points_mass_center
lidar_points_new = lidar_points - lidar_points_mass_center

print(f"world_points_mass_center: {world_points_mass_center}")
print(f"lidar_points_mass_center: {lidar_points_mass_center}")
H = np.dot(lidar_points_new.T, world_points_new)


# print(H)
U, sigma, VT = np.linalg.svd(H)

R = np.dot(VT.T, U.T)
if np.linalg.det(R) < 0:
	V = VT.T
	V[:, -1] = -1 * V[:, -1]
	R = np.dot(V, U.T) 
print()
# print("idjos\n", world_points_mass_center, '\n', world_points_mass_center.reshape(-1, 1))
# print(R)

T = world_points_mass_center.reshape(-1, 1) - np.dot(R, lidar_points_mass_center.reshape(-1, 1))

print(world_points)
print(lidar_points)
# print(world_points[0])
# print(lidar_points[0])
# print()
for i in range(4):
	loss = np.linalg.norm(world_points[i] - (np.dot(R, lidar_points[i].reshape(-1, 1)) + T).reshape(-1))
	print(f"loss {i}. {loss}")
print(f"R_1: \n{R}")
print(f"T_1: {T}")
print(f"C_w: {C_w}")
print("====================")

import numpy as np
from scipy.spatial.transform import Rotation


# Create a Rotation object from the rotation matrix
rotation = Rotation.from_matrix(R)

# Convert to Euler angles (in radians)
# The order 'xyz' can be changed to other orders such as 'zyx', 'yxz', etc.
euler_angles = rotation.as_euler('xyz', degrees=True)

print('Euler Angles (degrees):', euler_angles)

ans = input("collect lidar points? ")
if ans == 'y':
	directory_name = input("Directory Name: ")
	base_path = os.path.join("data_list", directory_name)
	lidar_points_path = os.path.join(base_path, "lidar_points.npy")
	world_points_path = os.path.join(base_path, "world_points.npy")
	R_path = os.path.join(base_path, "R.npy")
	T_path = os.path.join(base_path, "T.npy")

	if not os.path.exists(base_path):
		os.makedirs(base_path)

	np.save(lidar_points_path, lidar_points)
	np.save(world_points_path, world_points)
	np.save(R_path, R)
	np.save(T_path, T)
	print("done!")