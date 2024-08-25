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

		# sort_id = np.argsort(box_points[:, 0])
		# box_points = box_points[sort_id]
		# box_x1 = box_points[0]
		# box_x2 = box_points[-1]
		# width = np.linalg.norm(box_x1-box_x2)
		
		return (wall_rmse)**2  #+ (wall_rmse)**2
	
	coefs = [wall_point]

	bounds = [(-90, 90)]
	initial_guess = [0]
	# result = minimize(objective_function, initial_guess, args=coefs, tol=1e-5, bounds=bounds, method='trust-constr')
	result = minimize(objective_function, initial_guess, args=coefs, bounds=bounds, tol=1e-5, method='trust-constr')
	print("P: ", result.x)
	return result.x
