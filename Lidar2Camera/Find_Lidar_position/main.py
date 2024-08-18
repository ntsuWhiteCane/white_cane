import math

import cv2
import numpy as np

import newton_raphson
import read_pcd
import select_lidar_roi

threshold = 0.01

ind = 3 
data_index = str(ind).zfill(4)
lidar1_point_cloud_path = ".\\lidar\\PointClouds1\\" + data_index + ".pcd"
lidar2_point_cloud_path = ".\\lidar\\PointClouds2\\" + data_index + ".pcd"

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

points = read_pcd.read_pcd(lidar2_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
lidarRoi = select_lidar_roi.LidarRoi(points)
lidarRoi.show_figure()

tmp_mask1 = points[:, 0] >= lidarRoi.x_roi[0]
tmp_mask1 &= points[:, 0] <= lidarRoi.x_roi[1]
tmp_mask2 = points[:, 2] >= lidarRoi.z_roi[0]
tmp_mask2 &= points[:, 2] <= lidarRoi.z_roi[1]
tmp_mask = tmp_mask1 & tmp_mask2

lidar2_point = points[tmp_mask]

lidar2_image = lidarRoi.canvas[lidarRoi.start_y:lidarRoi.end_y, lidarRoi.start_x:lidarRoi.end_x]

cv2.namedWindow("lidar1", 0)
cv2.namedWindow("lidar2", 0)
cv2.imshow("lidar1", lidar1_image)
cv2.imshow("lidar2", lidar2_image)

print("lidar1.shape: ", lidar1_point.shape, "lidar2.shape: ", lidar2_point.shape)

cv2.waitKey(0)

lidar1_sort_id = np.argsort(lidar1_point[:, 0])
lidar1_sort = lidar1_point[lidar1_sort_id]


lidar2_sort_id = np.argsort(lidar2_point[:, 0])
lidar2_sort = lidar2_point[lidar2_sort_id]



cv2.destroyAllWindows()
lidarRoi = select_lidar_roi.LidarRoi(lidar1_sort)
lidar1 = lidarRoi.select_4_points()

cv2.namedWindow("4 points", 0)
cv2.imshow("4 points", lidarRoi.canvas)
cv2.waitKey(0) 

cv2.destroyAllWindows()
lidarRoi = select_lidar_roi.LidarRoi(lidar2_sort)
lidar2 = lidarRoi.select_4_points()

cv2.namedWindow("4 points", 0)
cv2.imshow("4 points", lidarRoi.canvas)
cv2.waitKey(0) 

print(lidar1)
print(lidar2)

lidar1_m1 = (lidar1[1, 1] - lidar1[0, 1]) / (lidar1[1, 0] - lidar1[0, 0])
lidar1_m2 = (lidar1[3, 1] - lidar1[2, 1]) / (lidar1[3, 0] - lidar1[2, 0])
lidar1_b1 = lidar1[1, 1] - lidar1_m1*lidar1[1, 0]
lidar1_b2 = lidar1[2, 1] - lidar1_m2*lidar1[2, 0]

lidar2_m1 = (lidar2[1, 1] - lidar2[0, 1]) / (lidar2[1, 0] - lidar2[0, 0])
lidar2_m2 = (lidar2[3, 1] - lidar2[2, 1]) / (lidar2[3, 0] - lidar2[2, 0])
lidar2_b1 = lidar2[1, 1] - lidar2_m1*lidar2[1, 0]
lidar2_b2 = lidar2[2, 1] - lidar2_m2*lidar2[2, 0]

lidar1_D_x = -1 * (lidar1_b2 - lidar1_b1) / (lidar1_m2 - lidar1_m1)
lidar1_D_z = lidar1_m1 * lidar1_D_x + lidar1_b1
print(f"lidar1_D_x: {lidar1_D_x}, lidar1_D_z: {lidar1_D_z}")


lidar2_D_x = -1 * (lidar2_b2 - lidar2_b1) / (lidar2_m2 - lidar2_m1)
lidar2_D_z = lidar2_m1 * lidar2_D_x + lidar2_b1
print(f"lidar2_D_x: {lidar2_D_x}, lidar2_D_z: {lidar1_D_z}")

A_lidar = np.array([lidar1[1, 0], lidar1[1, 1], 0])
B_lidar = np.array([lidar1[2, 0], lidar1[2, 1], 0])
D_lidar = np.array([lidar1_D_x, lidar1_D_z, 0])
C_lidar = np.array([0, 0, 0])
AB = np.linalg.norm(A_lidar-B_lidar)
BD = np.linalg.norm(B_lidar-D_lidar)
DA = np.linalg.norm(D_lidar-A_lidar)

k1 = np.power(AB, 2)
k2 = np.power(DA, 2)
k3 = np.power(BD, 2)

A_w = np.array([np.sqrt((k1+k2-k3)/2), 0, 0])
B_w = np.array([0, np.sqrt((k1-k2+k3)/2), 0])
D_w = np.array([0, 0, -1*np.sqrt((k2+k3-k1)/2)])
C_w = newton_raphson.find_it(A_point=A_w, B_point=B_w, D_point=D_w, len_a=np.linalg.norm(A_lidar), len_b=np.linalg.norm(B_lidar), len_d=np.linalg.norm(D_lidar))
# C_lidar = newton_raphson.newton_raphson(A_point=A_w, B_point=B_w, D_point=D_w, len_a=np.linalg.norm(A_lidar), len_b=np.linalg.norm(B_lidar), len_d=np.linalg.norm(D_lidar))
# print(C_w)
A_lidar = np.array([lidar1[1, 0], 0, lidar1[1, 1]])
B_lidar = np.array([lidar1[2, 0], 0, lidar1[2, 1]])
D_lidar = np.array([lidar1_D_x, 0, lidar1_D_z])
C_lidar = np.array([0, 0, 0])

world_points = np.stack((A_w, B_w, C_w, D_w))
lidar_points = np.stack((A_lidar, B_lidar, C_lidar, D_lidar))

# lidar_points = lidar_points - C_w

world_points_mass_center = (A_w + B_w + C_w + D_w)/4
lidar_points_mass_center = (A_lidar + B_lidar + C_lidar + D_lidar)/4

world_points_new = world_points - world_points_mass_center
lidar_points_new = lidar_points - lidar_points_mass_center
# print("hi")
print(f"world_points_mass_center: {world_points_mass_center}")
print(f"lidar_points_mass_center: {lidar_points_mass_center}")
H = np.dot(lidar_points_new.T, world_points_new)
# print(H)
U, sigma, VT = np.linalg.svd(H)

R = np.dot(VT.T, U.T)
print()
# print(R)

T = world_points_mass_center.T - np.dot(R, lidar_points_mass_center.T)
print(f"R_1: \n{R}")
print(f"T_1: {T}")
print(f"C_w: {C_w}")
print(world_points)
print(lidar_points)
print("====================")
######################################################################
A_lidar = np.array([lidar2[1, 0], lidar2[1, 1], 0])
B_lidar = np.array([lidar2[2, 0], lidar2[2, 1], 0])
D_lidar = np.array([lidar2_D_x, lidar2_D_z, 0])
C_lidar = np.array([0, 0, 0])
AB = np.linalg.norm(A_lidar-B_lidar)
BD = np.linalg.norm(B_lidar-D_lidar)
DA = np.linalg.norm(D_lidar-A_lidar)

k1 = np.power(AB, 2)
k2 = np.power(DA, 2)
k3 = np.power(BD, 2)

A_w = np.array([np.sqrt((k1+k2-k3)/2), 0, 0])
B_w = np.array([0, np.sqrt((k1-k2+k3)/2), 0])
D_w = np.array([0, 0, -1*np.sqrt((k2+k3-k1)/2)])
C_w = newton_raphson.find_it(A_point=A_w, B_point=B_w, D_point=D_w, len_a=np.linalg.norm(A_lidar), len_b=np.linalg.norm(B_lidar), len_d=np.linalg.norm(D_lidar))
# C_lidar = newton_raphson.newton_raphson(A_point=A_w, B_point=B_w, D_point=D_w, len_a=np.linalg.norm(A_lidar), len_b=np.linalg.norm(B_lidar), len_d=np.linalg.norm(D_lidar))
# print(C_w)
A_lidar = np.array([lidar2[1, 0], 0, lidar2[1, 1]])
B_lidar = np.array([lidar2[2, 0], 0, lidar2[2, 1]])
D_lidar = np.array([lidar2_D_x, 0, lidar2_D_z])
C_lidar = np.array([0, 0, 0])

world_points = np.stack((A_w, B_w, C_w, D_w))
lidar_points = np.stack((A_lidar, B_lidar, C_lidar, D_lidar))

world_points_mass_center = (A_w + B_w + C_w + D_w)/4
lidar_points_mass_center = (A_lidar + B_lidar + C_lidar + D_lidar)/4

world_points_new = world_points - world_points_mass_center
lidar_points_new = lidar_points - lidar_points_mass_center
print("hi")
# print(world_points)
# print(lidar_points_mass_center)
H = np.dot(lidar_points_new.T, world_points_new)
# print(f"dude: {lidar_points_new.T}")
# print(H)
U, sigma, VT = np.linalg.svd(H)

R_2 = np.dot(VT.T, U.T)
print()
# print(R_2)

T_2 = world_points_mass_center.T - np.dot(R_2, lidar_points_mass_center.T)
print(f"R_2: \n{R_2}")
print(f"T_2: {T_2}")
print(f"C_w: {C_w}")
print(world_points)
print(lidar_points)
print()

print(R)
print(T)
print("hi")
print(R_2)
print(T_2)
print("hihi")
print("R: ")
print(np.dot(R_2.T, R))
print("T: ")
print(np.dot(R.T, (T_2-T)))

