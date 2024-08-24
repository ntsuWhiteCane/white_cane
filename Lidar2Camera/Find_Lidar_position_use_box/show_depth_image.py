import cv2
import numpy as np

import read_pcd
import select_lidar_roi

zed_K = np.array([[3.506789914489589e+02, 0, 3.127677577917245e+02],
                  [0, 3.510410495486862e+02, 1.875433445883912e+02],
                  [0, 0, 1]])

ind = 2 #25 
data_index = str(ind).zfill(4)

zed_point_cloud_path = ".\\lidar_zed_valid\\Zed_Point_Clouds\\" + data_index + ".pcd"

points = read_pcd.read_pcd(zed_point_cloud_path)
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)

points = np.dot(points, zed_K.T)
mask = points[:, 2] <= 0
points = points[~mask]

d = points[:, 2]


for i in range(2):
    points[:,i] = points[:, i] / points[:, 2]

points[:, 2] = d

mask = points[:, 0] <= 0
mask |= points[:, 1] <= 0
mask |= points[:, 0] >= 640
mask |= points[:, 1] >= 360
mask |= points[:, 2] <= 0

points = points[~mask]
print(points.shape)
max_depth = np.max(points[:, 2])
min_depth = np.min(points[:, 2])
print(max_depth, min_depth)
points[:, 2] = (points[:, 2]-min_depth) / (max_depth - min_depth) * 255

canvas = np.zeros((360, 640), dtype=np.uint8)

for i in range(len(points)):
    canvas[int(points[i, 1]), int(points[i, 0])] = points[i, 2]
cv2.namedWindow("ii", 0)
cv2.imshow("ii", canvas)
cv2.waitKey(0)