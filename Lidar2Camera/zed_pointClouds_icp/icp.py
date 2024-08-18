import open3d as o3d
import copy
import math
import numpy as np

# Initialize functions
def draw_registration_result(source, target, transformation):
    """
    param: source - source point cloud
    param: target - target point cloud
    param: transformation - 4 X 4 homogeneous transformation matrix
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], zoom=0.4459, front=[0.9288, -0.2951, -0.2242], lookat=[1.6784, 2.0612, 1.4451], up=[-0.3402, -0.9189, -0.1996])

def find_nearest_neighbors(source_pc, target_pc, nearest_neigh_num):
    # Find the closest neighbor for each anchor point through KDTree
    point_cloud_tree = o3d.geometry.KDTreeFlann(source_pc)
    # Find nearest target_point neighbor index
    points_arr = []
    for point in target_pc.points:
        [_, idx, _] = point_cloud_tree.search_knn_vector_3d(point, nearest_neigh_num)
        points_arr.append(source_pc.points[idx[0]])
    return np.asarray(points_arr)


def icp(source, target):
    source.paint_uniform_color([0.5, 0.5, 0.5])
    target.paint_uniform_color([0, 0, 1])
    #source_points = np.asarray(source.points) # source_points is len()=198835x3 <--> 198835 points that have (x,y,z) val
    target_points = np.asarray(target.points)
    # Since there are more source_points than there are target_points, we know there is not
    # a perfect one-to-one correspondence match. Sometimes, many points will match to one point,
    # and other times, some points may not match at all.

    transform_matrix = np.asarray([[1.0, 0.0, 0.0, 0.075], [0.0, 1.0, 0.0, 0.295], [0.0, 0.0, 1.0, 0.4], [0.0, 0.0, 0.0, 1.0]])
    source = source.transform(transform_matrix)

    # While loop variables
    curr_iteration = 0
    cost_change_threshold = 0.001
    curr_cost = 1000
    prev_cost = 10000

    while (True):
        # 1. Find nearest neighbors
        new_source_points = find_nearest_neighbors(source, target, 1)

        # 2. Find point cloud centroids and their repositions
        source_centroid = np.mean(new_source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)
        source_repos = np.zeros_like(new_source_points)
        target_repos = np.zeros_like(target_points)
        source_repos = np.asarray([new_source_points[ind] - source_centroid for ind in range(len(new_source_points))])
        target_repos = np.asarray([target_points[ind] - target_centroid for ind in range(len(target_points))])

        # 3. Find correspondence between source and target point clouds
        cov_mat = target_repos.transpose() @ source_repos

        U, X, Vt = np.linalg.svd(cov_mat)
        R = U @ Vt
        t = target_centroid - R @ source_centroid
        t = np.reshape(t, (1,3))
        curr_cost = np.linalg.norm(target_repos - (R @ source_repos.T).T)
        print("Curr_cost=", curr_cost)
        if ((prev_cost - curr_cost) > cost_change_threshold):
            prev_cost = curr_cost
            transform_matrix = np.hstack((R, t.T))
            transform_matrix = np.vstack((transform_matrix, np.array([0, 0, 0, 1])))
            # If cost_change is acceptable, update source with new transformation matrix
            source = source.transform(transform_matrix)
            curr_iteration += 1
        else:
            break
    print("\nIteration=", curr_iteration)
    # Visualize final iteration and print out final variables
    draw_registration_result(source, target, transform_matrix)
    return transform_matrix

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
### PART B ###


idx = 1
data_index = str(idx).zfill(4)
zed_image_path = "Images\\" + data_index + ".png"
lidar_point_cloud_path = "PointClouds\\" + data_index + ".pcd"
zed_point_cloud_path = "ZED_Point_Clouds\\" + data_index + ".pcd"
depth_path = "Depth\\" + data_index + ".npy"


source = o3d.io.read_point_cloud(lidar_point_cloud_path)
target = o3d.io.read_point_cloud(zed_point_cloud_path)


points = read_pcd(lidar_point_cloud_path)

# points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
	
lidar_pcd = o3d.geometry.PointCloud()
lidar_pcd.points = o3d.utility.Vector3dVector(points)

points = read_pcd(zed_point_cloud_path)

# points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
zed_pcd = o3d.geometry.PointCloud()
zed_pcd.points = o3d.utility.Vector3dVector(points)

source = lidar_pcd
target = zed_pcd

print(source.points, target.points)
part_b = icp(source, target)
print(part_b)