import cv2
import math
import numpy as np

import select_camera_roi

transformation = np.load("transformation.npy", 'r')
# transformation = np.load("nice.npy", 'r')
print(transformation)

R = transformation[0:3, 0:3]
T = transformation[0:3, 3] * 1000

zed_K = np.array([[3.506789914489589e+02, 0, 3.127677577917245e+02],
                  [0, 3.510410495486862e+02, 1.875433445883912e+02],
                  [0, 0, 1]])
# R = np.array([[1,0,0],[0,1,0],[0,0,1]])
# T = np.array([75, 295, 400])
print("R:\n", R)
print("T:\n", T)

data_index = "0009"
zed_image_path = "Images\\" + data_index + ".png"
lidar_point_cloud_path = "PointClouds\\" + data_index + ".pcd"
zed_point_cloud_path = "ZED_Point_Clouds\\" + data_index + ".pcd"

zed_img = cv2.imread(zed_image_path, -1)


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

points = read_pcd(lidar_point_cloud_path) * 1000

points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1) 

print(points.shape)

pointsOnZED = np.dot(points, R) + T.T
# print(pointsOnZED.shape)

depth_mask = pointsOnZED[:, 2] > 0
pointsOnImage = np.dot(pointsOnZED[depth_mask], zed_K.T)
print(pointsOnImage[10])
for i in range(2):
	pointsOnImage[:, i] = pointsOnImage[:, i] / pointsOnImage[:, 2]
print(pointsOnImage.shape)
width_mask = pointsOnImage[:, 0] >= 0
width_mask = width_mask & (pointsOnImage[:, 0] < zed_img.shape[1])
height_mask = pointsOnImage[:, 1] >= 0 
height_mask = height_mask & (pointsOnImage[:, 1] < zed_img.shape[0])
projection_mask = width_mask & height_mask

pointsOnImage = pointsOnImage[projection_mask]
print(pointsOnImage.shape)

projection_img = zed_img
for i in range(pointsOnImage.shape[0]):
    projection_img = cv2.circle(zed_img, (int(pointsOnImage[i, 0]), int(pointsOnImage[i, 1])), 1, (255, 0, 0), -1)
	
cv2.namedWindow("zed", 0)
cv2.imshow("zed", projection_img)
cv2.waitKey(0)

points = read_pcd(zed_point_cloud_path) * 1000
points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)

proj = select_camera_roi.CameraRoi(zed_img, points, zed_K)
proj.start_x = 0
proj.end_x = zed_img.shape[1]
proj.start_y = 0
proj.end_y = zed_img.shape[0]

proj.computePoint()

ground_truth = proj.canvas
pic = (proj.canvas-np.min(proj.canvas))/(np.max(proj.canvas) - np.min(proj.canvas))
pic = 255*pic
pic = pic.astype(np.uint8)
mmin = np.min(proj.canvas)
print(mmin)

cv2.imshow("qq", pic)
cv2.waitKey(0)

sum = 0
count = 0
for i in range(pointsOnImage.shape[0]):
	if ground_truth[int(pointsOnImage[i, 1]), int(pointsOnImage[i, 0])] <= 0.1:
		continue
	sum += np.abs(pointsOnImage[i, 2] - ground_truth[int(pointsOnImage[i, 1]), int(pointsOnImage[i, 0])])/1000
	count += 1
print(sum ,count)
print(sum/count)