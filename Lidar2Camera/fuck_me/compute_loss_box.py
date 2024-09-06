import copy
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import newton_raphson
import read_pcd
import select_lidar_roi
import newton_raphson

max_ind = 221

def compute_R_imu2lidar(ypr):
	r = Rotation.from_euler("yxz", ypr)
	r = r.as_matrix()
	return r
def compute_R_lidar(ypr):
	r = Rotation.from_euler("zyx", ypr)
	r = r.as_matrix()
	return r
mean_depth_list = []
lidar2_x_list = []
t_z_list = []
box_1 = []
box_2 = []

random_list = np.arange(1, max_ind)
for i in range(random_list.shape[0]):
    num = np.random.randint(max_ind-1)
    tmp = random_list[i]
    random_list[i] = random_list[num]
    random_list[num] = tmp

for ind in range(1, max_ind):
	data_index = str(ind).zfill(4)
	tmp_index = str(ind-1).zfill(4)
	lidar1_point_cloud_path = ".\\box\\lidar1\\box" + tmp_index + ".npy"
	lidar2_point_cloud_path = ".\\box\\lidar2\\box" + tmp_index + ".npy"
	image_path = ".\\box_data\\Images\\" + data_index + ".png"
	imu_path = ".\\box_data\\Imu\\" + data_index + ".npy"

	imu = np.load(imu_path)
	points = np.load(lidar1_point_cloud_path)
	points_sort_id = np.argsort(points[:, 0])
	lidar1_wall = points[points_sort_id]
 
	pitch = newton_raphson.find_pitch(lidar1_wall)
	pitch = pitch[0]*np.pi/180
	# imu system is z y x, but in lidar system is -y -x z
	ypr = np.array([-1*pitch, -1*imu[1], imu[2]])

	lidar1_wall = np.dot(lidar1_wall, compute_R_imu2lidar(ypr=ypr))

	points = np.load(lidar2_point_cloud_path)
	points_sort_id = np.argsort(points[:, 0])
	lidar2_wall = points[points_sort_id]
	# select_lidar_roi.LidarRoi(lidar1_wall).show_points(lidar1_wall)
	# select_lidar_roi.LidarRoi(lidar2_wall).show_points(lidar2_wall)	
	lidar2_wall = np.dot(lidar2_wall, compute_R_imu2lidar(ypr=ypr))
	box_x_1 = [list(lidar1_wall[np.argmin(lidar1_wall[:, 0])]), list(lidar1_wall[np.argmax(lidar1_wall[:, 0])]) ]
	box_x_2 = [list(lidar2_wall[np.argmin(lidar2_wall[:, 0])]), list(lidar2_wall[np.argmax(lidar2_wall[:, 0])])]
	# print(box_x_1)
	box_1.append(box_x_1)
	box_2.append(box_x_2)

train_data_x = []
train_data_y = []
test_data_x = []
test_data_y = []
for i in range(50):
	# print(random_list[i])
	train_data_x.append(box_1[random_list[i]-1])
	train_data_y.append(box_2[random_list[i]-1])
for i in range(50, max_ind-1):
	# print(random_list[i])
	test_data_x.append(box_1[random_list[i]-1])
	test_data_y.append(box_2[random_list[i]-1])

# y, p, r, t_z = newton_raphson.wall_fit(lidar1_wall, lidar2_wall, [0, 0, 0, 0.13])
# print(train_data_x[0])
(y, p, r, t_x), error_list = newton_raphson.box_fit(train_data_x, train_data_y, [0, 0, 0, 0])
print(y, p, r, t_x)
error_list = np.array(error_list)
plt.plot(error_list[:, 0], error_list[:, 1])
plt.xlabel("iteration")
plt.ylabel("error")
plt.show()
# for i in range(20):
# 	print(test_data_x[i][0][0],test_data_y[i][0][0])

for i in range(50, max_ind-1):
	lidar1_box = np.array(box_1[random_list[i]-1])
	lidar2_box = np.array(box_2[random_list[i]-1])
	R_3 = np.array([[9.99855187e-01,  1.70091366e-02, -5.42651288e-04], [-1.67885444e-02, 9.80668225e-01, -1.94956348e-01], [-2.78387828e-03, 1.94937226e-01, 9.80811770e-01]])
	# R_3 = np.identity(3)
	T_3 = np.array([[-0.003917707], [0.35186587], [0.17140681]])
	lidar2_box = np.dot(lidar2_box, compute_R_lidar((y, p, r)).T) + np.array([t_x, 0, 0])
	# lidar2_box = np.dot(lidar2_box, R_3.T) + T_3.reshape(-1)

	mean_depth = np.mean(lidar1_box[:, 2])
	mean_depth_list.append(mean_depth)
	# print(lidar1_box[0, 0] - lidar2_box[1, 0])
	lidar2_x_list.append(np.abs(lidar2_box[0, 0]-lidar1_box[0, 0]) + np.abs(lidar2_box[1, 0]-lidar1_box[1, 0]))
# print(mean_depth_list[0])
# print(t_z_list[0])
mean_depth_list = np.array(mean_depth_list)


plt.figure()
plt.scatter(mean_depth_list, np.array(lidar2_x_list)/mean_depth_list * 100)
# plt.figure()
plt.xlabel("depth", fontsize=20)
plt.ylabel("%" + "error",fontsize=20)
plt.show()
plt.close()

print(np.mean(lidar2_x_list))

boxplot_x = [i for i in np.linspace(np.min(mean_depth_list), np.max(mean_depth_list), 5)]
boxplot_x = [i*0.5 for i in range(1, 6, 1)]
y_list = []
for i in range(len(boxplot_x)):
	th = (np.max(mean_depth_list)-np.min(mean_depth_list))/(len(boxplot_x))
	# print(th)
	mask = mean_depth_list > np.min(mean_depth_list) + i*th
	mask &= mean_depth_list < np.min(mean_depth_list) + (i+1)*th
	y_list.append((np.array(lidar2_x_list)[mask])/mean_depth_list[mask] * 100)
print(boxplot_x)
plt.figure()
plt.boxplot(y_list, positions=np.round(boxplot_x,3), sym="")
# plt.plot(plot_x, diff_mean)
plt.xlabel("Ref(m)", fontsize=20)
plt.ylabel("%error", fontsize=20)
# plt.xlim((0, 7))
# plt.ylim((0, 16))
plt.title("x axis", fontsize=20)
plt.show()
