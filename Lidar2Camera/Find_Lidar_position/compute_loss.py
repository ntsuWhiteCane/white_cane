import cv2
import numpy as np

import read_pcd
import select_lidar_roi
'''
R = np.array([
    [0.99964178,  0.02266534, -0.0142334 ],
    [-0.02161195,  0.99729556,  0.07024594],
    [ 0.01578705, -0.06991316,  0.99742815]])
T = np.array([[-0.01533125], [0.14097775], [0.15504766]])
'''

R = np.array([
    [0.99736369,  -0.06001906, 0.04078455],
    [0.06098834,  0.99787457,  -0.02295134],
    [-0.03932034 , 0.02537822,  0.99890433]])
R = np.array([
    [1,  0, 0],
    [0,  1,  0],
    [0 , 0,  1]])
T = np.array([[0.01091015], [-0.11989352], [-0.12963439]])

ind = 3
data_index = str(ind).zfill(4)
lidar1_point_cloud_path = ".\\lidar_valid\\PointClouds1\\" + data_index + ".pcd"
lidar2_point_cloud_path = ".\\lidar_valid\\PointClouds2\\" + data_index + ".pcd"

points = read_pcd.read_pcd(lidar1_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
# tmp = points[0].reshape(3,1)
# print(tmp)
# print(R)
# print(T)
# print(np.dot(R, tmp) + T)
for i in range(len(points)):
    # p = points[i].reshape(3, 1)
    # p = np.dot(R.T, p - T)
    p = points[i].reshape(3, 1)
    p = np.dot(R, p) + T
    # print(p)
    points[i] = p.T
# points = np.dot(points, R)
# points = points - T.T

lidarRoi = select_lidar_roi.LidarRoi(points)
lidarRoi.show_figure()

tmp_mask1 = points[:, 0] >= lidarRoi.x_roi[0]
tmp_mask1 &= points[:, 0] <= lidarRoi.x_roi[1]
tmp_mask2 = points[:, 2] >= lidarRoi.z_roi[0]
tmp_mask2 &= points[:, 2] <= lidarRoi.z_roi[1]
tmp_mask = tmp_mask1 & tmp_mask2
lidar1_point = points[tmp_mask]

# print(lidar1_point)

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
# print(lidar2_point)
lidar2_image = lidarRoi.canvas[lidarRoi.start_y:lidarRoi.end_y, lidarRoi.start_x:lidarRoi.end_x]

cv2.namedWindow("lidar1", 0)
cv2.namedWindow("lidar2", 0)
cv2.imshow("lidar1", lidar1_image)
cv2.imshow("lidar2", lidar2_image)

cv2.waitKey(0)

print(lidar1_point.shape, lidar2_point.shape)

# norm1 = np.linalg.norm(lidar1_point[:, 2], axis=-1)
# norm2 = np.linalg.norm(lidar2_point[:, 2], axis=-1)
# print(lidar1_point[:, 2].reshape(-1, 1))
print(f"max: lidar1: {np.max(lidar1_point[:, 2])}, lidar2: {np.max(lidar2_point[:, 2])}")
print(f"min lidar1: {np.min(lidar1_point[:, 2])}, lidar2: {np.min(lidar2_point[:, 2])}")
print(f"mean: lidar1: {np.mean(lidar1_point[:, 2])}, lidar2: {np.mean(lidar2_point[:, 2])}")

lidar1_x = lidar1_point[:, 0]
lidar2_x = lidar1_point[:, 0]
print(f"lidar1_x: {np.max(lidar1_x) - np.min(lidar1_x)}, lidar2_x: {np.max(lidar2_x)-np.min(lidar2_x)}")
print(f"lidar1_min_x: {np.min(lidar1_x)}, lidar1_max_x: {np.max(lidar1_x)}")
print(f"lidar2_min_x: {np.min(lidar2_x)}, lidar2_max_x: {np.max(lidar2_x)}")
print(lidar1_point[0], lidar2_point[0])


# import open3d as o3d

# lidar_point = np.vstack((lidar1_point, lidar2_point))

# lidar_pcd = o3d.geometry.PointCloud()
# lidar_pcd.points = o3d.utility.Vector3dVector(lidar_point)

# o3d.visualization.draw_geometries(geometry_list = [lidar_pcd],  window_name="point_cloud")