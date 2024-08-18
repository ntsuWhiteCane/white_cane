import matplotlib.pyplot as plt
import math
import numpy as np
import open3d as o3d
# from mpl_toolkits.mplot3d import Axes3D

data_index = "0001"
zed_image_path = "Images\\" + data_index + ".png"
lidar_point_cloud_path = "PointClouds\\" + data_index + ".pcd"
zed_point_cloud_path = "ZED_Point_Clouds\\" + data_index + ".pcd"
print(float("nan") )
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

print("load")
points = read_pcd(zed_point_cloud_path)

print(f"read point clouds by my function: {points.shape}\n")


zed_pcd = o3d.io.read_point_cloud(zed_point_cloud_path)

lidar_pcd = o3d.io.read_point_cloud(lidar_point_cloud_path)
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.075],
						 [0.0, 1.0, 0.0, 0.295],
						 [0.0, 0.0, 1.0, 0.4],
						 [0.0, 0.0, 0.0, 1.0]])



for i in range(11, 12):
	data_index = str(i).zfill(4)
	zed_image_path = "Images\\" + data_index + ".png"
	lidar_point_cloud_path = "PointClouds\\" + data_index + ".pcd"
	zed_point_cloud_path = "ZED_Point_Clouds\\" + data_index + ".pcd"
	depth_path = "Depth\\" + data_index + ".npy"
 
	points = read_pcd(lidar_point_cloud_path)

	points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
	
	lidar_pcd = o3d.geometry.PointCloud()
	lidar_pcd.points = o3d.utility.Vector3dVector(points)

	points = read_pcd(zed_point_cloud_path)

	points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
	zed_pcd = o3d.geometry.PointCloud()
	zed_pcd.points = o3d.utility.Vector3dVector(points)

	# print(type(zed_pcd), type(lidar_pcd))
	# [-56.388641564943384],
	# [-1.755408764448477e+02],
	# [-1.943824622517792e+02]

	depth = np.load(depth_path)

	# print(f"trans_init: \n{trans_init}")
	reg_p2p = o3d.pipelines.registration.registration_icp(lidar_pcd, zed_pcd, 0.2, trans_init, 
														o3d.pipelines.registration.TransformationEstimationPointToPoint(),
														o3d.pipelines.registration.ICPConvergenceCriteria(100000))
	trans_init = reg_p2p.transformation 

print(reg_p2p)
print(reg_p2p.transformation, "\n")

print("zed pcd:", zed_pcd)
print("lidar pcd:", lidar_pcd)
print()
# pcd_filtered = pcd.uniform_down_sample(every_k_points=3)


o3d.visualization.draw_geometries(geometry_list = [zed_pcd],  window_name="point_cloud")

print(type(reg_p2p.transformation))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(out_arr[:, 0], out_arr[:, 1], out_arr[:, 2], c='b', marker='o')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# ax.set_title('3D Point Cloud')

# plt.show()


np.save("transformation.npy", reg_p2p.transformation)