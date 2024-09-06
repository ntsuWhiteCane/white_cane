import math

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

def find_pitch(wall_point) -> np.ndarray:
	def objective_function(coords, coefs):
		pitch = coords[0]
		
		pitch = pitch*np.pi/180

		rotation_matrix = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])

		wall_points = coefs[0]

		wall_points = np.dot(wall_points, rotation_matrix.T)

		wall_depth = wall_points[:, 2]
		wall_mean_depth = np.mean(wall_depth)

		wall_square_error = np.power(wall_depth-wall_mean_depth, 2)

		wall_rmse = np.sqrt(np.mean(wall_square_error))

		
		return (wall_rmse)**2  
	
	coefs = [wall_point]

	bounds = [(-90, 90)]
	initial_guess = [0]
	# result = minimize(objective_function, initial_guess, args=coefs, tol=1e-5, bounds=bounds, method='trust-constr')
	result = minimize(objective_function, initial_guess, args=coefs, bounds=bounds, tol=1e-5, method='trust-constr')
	# print("P: ", result.x)
	return result.x

def wall_fit(lidar1_points, lidar2_points, initial_value):
	iteration = 1
	error_list = []
	def objective_function(coords, coefs):
		yaw = coords[0]
		pitch = coords[1]
		roll = coords[2]
		t_z = coords[3]
		r = Rotation.from_euler("zyx", [yaw, pitch, roll])
		r = r.as_matrix()
		wall_square_error = 0
		count = 0
		for i in range(len(lidar1_points)):
			# print(lidar1_points[i])
			lidar1_wall_points = lidar1_points[i]
			lidar2_wall_points = lidar2_points[i]

			lidar1_wall_depth = lidar1_wall_points[:, 2]
			lidar1_wall_mean_depth = np.mean(lidar1_wall_depth)

			lidar2_wall_points = np.dot(lidar2_wall_points, r.T) + np.array([0, 0, t_z])

			lidar2_wall_depth = lidar2_wall_points[:, 2]
			lidar2_wall_mean_depth = np.mean(lidar2_wall_depth)
			wall_square_error += np.mean(np.power(lidar1_wall_mean_depth-lidar2_wall_mean_depth, 2))
			count += 1
		# print(wall_square_error)

		wall_rmse = np.sqrt(wall_square_error/count)
		nonlocal iteration
		error_list.append([iteration, wall_rmse])
		iteration += 1
		return (wall_rmse)**2  #+ (wall_rmse)**2
	
	# coefs = [lidar1_points, lidar2_points]
	coefs = [iteration]
	bounds = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (None, None)]
	initial_guess = initial_value 
	# result = minimize(objective_function, initial_guess, args=coefs, tol=1e-5, bounds=bounds, method='trust-constr')
	result = minimize(objective_function, initial_guess, args=coefs, bounds=bounds, tol=1e-5, method='trust-constr')
	# print("P: ", result.x)
	return result.x, error_list

def box_fit(lidar1_points, lidar2_points, initial_value):
	iteration = 1
	error_list = []
	def objective_function(coords, coefs):
		yaw = coords[0]
		pitch = coords[1]
		roll = coords[2]
		t_x = coords[3]
		r = Rotation.from_euler("zyx", [yaw, pitch, roll])
		r = r.as_matrix()
		box_start_error = 0
		box_end_error = 0
		box_square_error = 0
		count = 0
		for i in range(len(lidar1_points)):
			# print(lidar1_points[i])
			lidar1_box_points = np.array(lidar1_points[i])
			lidar2_box_points = np.array(lidar2_points[i])
			# print(lidar2_box_points)
			lidar2_box_points = np.dot(lidar2_box_points, r.T) + np.array([t_x, 0, 0])

			box_start_error += np.power(lidar1_box_points[0, 0] - lidar2_box_points[0, 0], 2)
			box_end_error += np.power(lidar1_box_points[-1, 0] - lidar2_box_points[-1, 0], 2)
			
			count += 1
		box_start_rmse = np.sqrt(box_start_error/count)
		box_end_rmse = np.sqrt(box_end_error/count)
		nonlocal iteration
		error_list.append([iteration, box_start_rmse + box_end_rmse])
		iteration += 1
		return (box_start_rmse)**2 + (box_end_rmse)**2  #+ (wall_rmse)**2
	
	# coefs = [lidar1_points, lidar2_points]
	coefs = [iteration]
	bounds = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (None, None)]
	initial_guess = initial_value 
	# result = minimize(objective_function, initial_guess, args=coefs, tol=1e-5, bounds=bounds, method='trust-constr')
	result = minimize(objective_function, initial_guess, args=coefs, bounds=bounds, tol=1e-5, method='trust-constr')
	# print("P: ", result.x)
	return result.x, error_list


def tilt_box_fit(lidar1_points, lidar2_points, initial_value):
	iteration = 1
	error_list = []
	def objective_function(coords, coefs):
		yaw = coords[0]
		pitch = coords[1]
		roll = coords[2]
		t_y = coords[3]
		r = Rotation.from_euler("zyx", [yaw, pitch, roll])
		r = r.as_matrix()
		box_start_error = 0
		box_end_error = 0
		box_square_error = 0
		count = 0
		for i in range(len(lidar1_points)):
			# print(lidar1_points[i])
			lidar1_box_points = np.array(lidar1_points[i])
			lidar2_box_points = np.array(lidar2_points[i])
			# print(lidar2_box_points)
			lidar2_box_points = np.dot(lidar2_box_points, r.T) + np.array([0, t_y, 0])

			box_start_error += np.power((lidar2_box_points[0, 0] - lidar1_box_points[0, 0]) / t_y - np.tan(151*np.pi/180), 2)
			# box_end_error += np.power(lidar1_box_points[-1, 0] - lidar2_box_points[-1, 0], 2)
			
			count += 1
		box_start_rmse = np.sqrt(box_start_error/count)
		# box_end_rmse = np.sqrt(box_end_error/count)
		nonlocal iteration
		error_list.append([iteration, box_start_rmse])
		iteration += 1
		return (box_start_rmse)**2  #+ (wall_rmse)**2
	
	# coefs = [lidar1_points, lidar2_points]
	coefs = [iteration]
	bounds = [(-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (-np.pi/2, np.pi/2), (0.01, None)]
	initial_guess = initial_value 
	# result = minimize(objective_function, initial_guess, args=coefs, tol=1e-5, bounds=bounds, method='trust-constr')
	result = minimize(objective_function, initial_guess, args=coefs, bounds=bounds, tol=1e-5, method='trust-constr')
	# print("P: ", result.x)
	return result.x, error_list