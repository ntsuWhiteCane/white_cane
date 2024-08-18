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

zed_K = np.array([[3.506789914489589e+02, 0, 3.127677577917245e+02],
                  [0, 3.510410495486862e+02, 1.875433445883912e+02],
                  [0, 0, 1]])

zed_R = np.array([[0.9978751, -0.05798409, 0.02971761],
                  [0.0550915, 0.99438751, 0.09032394],
                  [-0.03478817, -0.08849482, 0.99546896]])
zed_T = np.array([[-0.0668869], [-0.29390763], [-0.30338202]])
# zed_R = np.identity(3)
# zed_T = np.array([[-0.0], [-0.], [-0.]])
R = np.array([
    [1,  0, 0],
    [0,  1,  0],
    [0 , 0,  1]])
T = np.array([[0.01091015], [-0.11989352], [-0.12963439]])

ind = 25 #25 
data_index = str(ind).zfill(4)
lidar1_point_cloud_path = ".\\lidar_zed_valid\\PointClouds1\\" + data_index + ".pcd"
lidar2_point_cloud_path = ".\\lidar_zed_valid\\PointClouds2\\" + data_index + ".pcd"
zed_point_cloud_path = ".\\lidar_zed_valid\\Zed_Point_Clouds\\" + data_index + ".pcd"
image_path = ".\\lidar_zed_valid\\Images\\" + data_index + ".png"
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
norm1 = np.linalg.norm(lidar1_point, axis=-1)
norm2 = np.linalg.norm(lidar2_point, axis=-1)

print(f"max: lidar1: {np.max(norm1)}, lidar2: {np.max(norm2)}")
print(f"mean: lidar1: {np.mean(norm1)}, lidar2: {np.mean(norm2)}")

lidar1_x = lidar1_point[:, 0]
lidar2_x = lidar1_point[:, 0]
print(f"lidar1_x: {np.max(lidar1_x) - np.min(lidar1_x)}, lidar2_x: {np.max(lidar2_x)-np.min(lidar2_x)}")
print(lidar1_point[0], lidar2_point[0])
import open3d as o3d

lidar_point = np.vstack((lidar1_point, lidar2_point))

lidar_pcd = o3d.geometry.PointCloud()
lidar_pcd.points = o3d.utility.Vector3dVector(lidar_point)


points = read_pcd.read_pcd(zed_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)

points = np.dot(points, zed_R.T) + zed_T.T

print(points.shape)

# o3d.visualization.draw_geometries(geometry_list = [lidar_pcd],  window_name="point_cloud")
zed_pcd = o3d.geometry.PointCloud()
zed_pcd.points = o3d.utility.Vector3dVector(points)

trans_init = np.asarray([[1, 0, 0, 0],
						 [0, 1, 0, 0],
						 [0, 0, 1, 0],
						 [0, 0, 0, 1]])
reg_p2p = o3d.pipelines.registration.registration_icp(lidar_pcd, zed_pcd, 0.1, trans_init, 
														o3d.pipelines.registration.TransformationEstimationPointToPoint(),
														o3d.pipelines.registration.ICPConvergenceCriteria(100))
print(reg_p2p)
print(reg_p2p.transformation, "\n")

print("zed pcd:", zed_pcd)
print("lidar pcd:", lidar_pcd)
print()
# pcd_filtered = pcd.uniform_down_sample(every_k_points=3)

zed_filter = zed_pcd.uniform_down_sample(every_k_points=50)
o3d.visualization.draw_geometries(geometry_list = [zed_filter, lidar_pcd],  window_name="point_cloud")
# o3d.visualization.draw_geometries(geometry_list = [lidar_pcd],  window_name="point_cloud")

lidar = lidar_point - zed_T.T
lidar = np.dot(lidar, zed_R)
lidar = np.dot(lidar, zed_K.T)

mask = lidar[:, 2] > 0
lidar = lidar[mask]

for i in range(2):
    lidar[:, i] = lidar[:, i] / lidar[:, 2]
mask = lidar[:, 0] <= 0
mask |= lidar[:, 1] <= 0
mask |= lidar[:, 0] >= 640
mask |= lidar[:, 1] >= 360
mask |= lidar[:, 2] <= 0

lidar = lidar[~mask]
print(lidar.shape)
img = cv2.imread(image_path, -1)

for i in range(len(lidar)):
    cv2.circle(img, (int(lidar[i, 0]), int(lidar[i, 1])), 2, (0, 0, 255), -1)
cv2.imshow("hi", img)
cv2.waitKey(0)
cv2.destroyAllWindows()