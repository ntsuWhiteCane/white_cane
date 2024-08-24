import math

import numpy as np

def newton_raphson(A_point, B_point, D_point, len_a, len_b, len_d):
	C_point = np.array([0.62133461, 0.63342518, 0.41925524])
	def condition_1():
		n = np.linalg.norm(C_point-A_point) 
		return np.power(n, 2) - (len_a ** 2)
	def condition_2():
		n = np.linalg.norm(C_point-B_point) 
		return np.power(n, 2) - (len_b ** 2)
	def condition_3():
		n = np.linalg.norm(C_point-D_point) 
		return np.power(n, 2) - (len_d ** 2)
	
	def jac():
		ret = np.zeros((3, 3))
		ret[0, 0] = 2*(C_point[0] - A_point[0])
		ret[0, 1] = 2*(C_point[1] - A_point[1])
		ret[0, 2] = 2*(C_point[2] - A_point[2])
		
		ret[1, 0] = 2*(C_point[0] - B_point[0])
		ret[1, 1] = 2*(C_point[1] - B_point[1])
		ret[1, 2] = 2*(C_point[2] - B_point[2])

		ret[2, 0] = 2*(C_point[0] - D_point[0])
		ret[2, 1] = 2*(C_point[1] - D_point[1])
		ret[2, 2] = 2*(C_point[2] - D_point[2])

		return ret
	flag = False
	f = np.array([[condition_1()], [condition_2()], [condition_3()]])
	for _ in range(3):
		# print(_)
		flag = True
		for i in range(len(f)):
			if abs(f[i,0]) > 0.001:
				flag = False
				break
		if flag:
			print("find it")
			break
		tmp_c = C_point.reshape((3, -1))
		
		jac_mat = jac()
		inc_jac = np.linalg.inv(jac_mat)
		tmp_c = tmp_c - np.dot(inc_jac, f)

		C_point[0] = tmp_c[0, 0]
		C_point[1] = tmp_c[1, 0]
		C_point[2] = tmp_c[2, 0]
		# print(tmp_c)
		# print(jac_mat)

	print(f"{condition_1()}, {condition_2()}, {condition_3()}")
	return C_point
from scipy.optimize import minimize
initial_guess = np.array([2, 2, 2])  # 

def find_it(A_point, B_point, D_point, len_a, len_b, len_d) -> np.ndarray:
	def objective_function(coords, coefs):
		x_C, y_C, z_C = coords
		A_point = coefs[0]
		B_point = coefs[1]
		D_point = coefs[2]
		dist = coefs[3]
		return (np.sqrt((x_C - A_point[0])**2 + (y_C - A_point[1])**2 + (z_C - A_point[2])**2) - dist[0])**2 + \
			(np.sqrt((x_C - B_point[0])**2 + (y_C - B_point[1])**2 + (z_C - B_point[2])**2) - dist[1])**2 + \
			(np.sqrt((x_C - D_point[0])**2 + (y_C - D_point[1])**2 + (z_C - D_point[2])**2) - dist[2])**2
	dist_arr = np.array([len_a, len_b, len_d])
	coefs = np.stack((A_point, B_point, D_point, dist_arr))
	bounds = [(None, None), (None, None), (None, None)]

	result = minimize(objective_function, initial_guess, args=coefs, tol=1e-5, bounds=bounds, method='trust-constr')
	print("C_w: ", result.x)
	return result.x
