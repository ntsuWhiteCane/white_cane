import cv2
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
import math
import copy
warnings.simplefilter(action='ignore', category=FutureWarning)

def kl_divergence(p, q):
	"""
	compute DL-Divergence
	p: observe probability
	q: reference probability divergence
	"""
	return entropy(p, q)

def check_uniform_distribution(data, bins=100):
	"""
	is uniform
	data: data
	bins: interval amount
	"""
	# get histogram
	observed_frequencies, _ = np.histogram(data, bins=bins, density=True)
	
	# compute reference distribution(uniform distribution)
	expected_frequencies = np.ones(bins) / bins
	
	# compute kl-divergence
	divergence = kl_divergence(observed_frequencies, expected_frequencies)
	
	return divergence
R = np.array([ [0.999943778871401, -0.007078039364406, 0.007895597199610],
			   [0.006683659654493, 0.998781109540383,   0.048904232116489],
			   [-0.008232119411517, -0.048848711180915,  0.998772264145315]])


T = np.array([ [-56.388641564943384],
			   [-1.755408764448477e+02],
			   [-1.943824622517792e+02]])
		
A = [[ 0.9994,  0.0134, 0.0314, -72.0912 ],
	 [-0.0147,  0.9991, 0.0399, -167.9426],
	 [-0.0309, -0.0403, 0.9987, -225.8460],
	 [	  0,	   0,	  0,		  1]]

zed_K = np.array([[7.059569895868153e+02,	0,	6.263567802746147e+02],
			   [	0,		 7.065462096249616e+02,	 3.765157523308447e+02],
			   [	0,		 0,		 1.0000]])

mono_K = np.array([[6.786368808170881e+02, 0,		6.440856249983478e+02],
				   [0,		6.780005400396101e+02, 3.474849689419397e+02],
				   [0,		0,		1.0000]])


inv_zed_K = np.linalg.inv(zed_K)
print("k*inv_k: \n", np.dot(zed_K, inv_zed_K))
print("inv_zed_K: \n", inv_zed_K)
inv_R = np.linalg.inv(R)


detect = False
start_x, start_y, end_x, end_y = 0, 0, 0, 0
mono_start_x, mono_start_y, mono_end_x, mono_end_y = 0, 0, 0, 0
zed_start_x, zed_start_y, zed_end_x, zed_end_y = 0, 0, 0, 0
alpha = 2
mono_alpha = 1.2
beta = 0

def select_roi(event, x, y, flags, param):
	global start_x, start_y, end_x, end_y
	global detect
	if event == cv2.EVENT_LBUTTONDOWN:
		start_x, start_y = x, y

	elif event == cv2.EVENT_LBUTTONUP:
		end_x, end_y = x, y
		cv2.destroyAllWindows()
		detect = True

def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
	look_up_table = (pow(np.arange(256) / 255, gamma) * 255).astype(np.uint8).reshape((1, 256))
					 
	res = cv2.LUT(img, look_up_table) #LUT，change explosure by look up table
	return res


def reprojection(inputPose) -> np.ndarray:
	pass
s = 0
n = 0
err_list = []
dis_list = []
depth_list = []
interval_depth_count = np.zeros(10)
interval_depth_value = np.zeros(10)

j = 0

for _ in range(13):
	# if _ == 10:
	#	continue

	if _ >= 9:
		mono_alpha = 1.1
		alpha = 1.5
	else:
		mono_alpha = 1.2
		alpha = 2

	mono_path = ".\\test_mono_ud\\mono" + str(_) + ".png"
	zed_path = ".\\test_zed\\zed_left" + str(_) + ".png"
	depth_path = ".\\test_depth\\depth" + str(_) + ".npy"
	depth_img_path = ".\\test_depth_image\\depth_image" + str(_) + ".png"
	depth = np.load(depth_path)

	mono_img = cv2.imread(mono_path, 0)
	zed_img = cv2.imread(zed_path, 0)
	depth_img = cv2.imread(depth_img_path, -1)

	mono_img = cv2.convertScaleAbs(mono_img, alpha=mono_alpha, beta=beta)
	zed_img = cv2.convertScaleAbs(zed_img, alpha=alpha, beta=beta)
	board_size = (7, 8)

	corner = np.array([1])
	mono_found, mono_corner = cv2.findChessboardCorners(mono_img, board_size, corners=corner, flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
	zed_found, zed_corner = cv2.findChessboardCorners(zed_img, board_size, corner, cv2.CALIB_CB_ADAPTIVE_THRESH)
	while mono_found == 0:
		cv2.namedWindow('mono', 0)
		cv2.setMouseCallback('mono', select_roi)
		cv2.imshow('mono', mono_img)
		cv2.waitKey(0)
		if detect:
			mono_found, mono_corner = cv2.findChessboardCorners(mono_img[start_y:end_y, start_x:end_x], board_size, corners=corner, flags=cv2.CALIB_CB_ADAPTIVE_THRESH) 
			if mono_found == True:
				mono_corner += np.array([start_x, start_y])
				mono_start_x, mono_start_y, mono_end_x, mono_end_y = start_x, start_y, end_x, end_y
	cv2.destroyAllWindows()
	while zed_found == 0:
		cv2.namedWindow('zed', 0)
		cv2.setMouseCallback('zed', select_roi)
		cv2.imshow('zed', zed_img)
		cv2.waitKey(0)
		if detect:
			zed_found, zed_corner = cv2.findChessboardCorners(zed_img[start_y:end_y, start_x:end_x], board_size, corners=corner, flags=cv2.CALIB_CB_ADAPTIVE_THRESH) 
			if zed_found == True:
				zed_corner += np.array([start_x, start_y])
				zed_start_x, zed_start_y, zed_end_x, zed_end_y = start_x, start_y, end_x, end_y
	cv2.destroyAllWindows()

	if mono_found != 0 and zed_found != 0:
		

		winSize = (10, 10)
		zeroZone = (-1, -1)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.001)
		mono_corner_subpixel = cv2.cornerSubPix(mono_img, mono_corner, winSize, zeroZone, criteria)
		zed_corner_subpixel = cv2.cornerSubPix(zed_img, zed_corner, winSize, zeroZone, criteria)

		mono_corner_subpixel = mono_corner_subpixel.reshape(-1, 2)
		zed_corner_subpixel = zed_corner_subpixel.reshape(-1, 2)
		
		p1 = zed_corner_subpixel[0]
		p2 = zed_corner_subpixel[-1]

		start_p = np.array(( min(p1[0], p2[0]),min(p1[1], p2[1]) )).astype(int)
		end_p = np.array(( max(p1[0], p2[0]),max(p1[1], p2[1]) )).astype(int)
		
		d_roi = depth[start_p[1]:end_p[1], start_p[0]:end_p[0]]
		d_roi = d_roi[np.where((~np.isnan(d_roi)) & (d_roi > 0))]
		
		d_mean = np.mean(d_roi)
		d_stdDev = np.std(d_roi)
		print(check_uniform_distribution(d_roi))
		print(_, ":", mono_corner_subpixel.shape)
		print("depth roi mean: ", d_mean, ";", "depth roi standard deviation: ", d_stdDev)

		safe_flag = True if (d_stdDev/d_mean < 0.05 or check_uniform_distribution(d_roi) < 0.2) else False
		print(f"is easy? {safe_flag}")
		if not safe_flag:
			kmeans = KMeans(n_clusters=4)
			kmeans.fit(d_roi.reshape(-1, 1))
			# print(d_roi.reshape(-1, 1))
			labels = kmeans.labels_
			unique_labels, labels_counts = np.unique(labels, return_counts=True)
			largest_cluster_label = unique_labels[np.argmax(labels_counts)]
			print(largest_cluster_label)
		
		pos_list = []
		index_list = []
		sum = 0
		for i in range(zed_corner_subpixel.shape[0]):
			pos = zed_corner_subpixel[i]
			d = depth[int(pos[1]), int(pos[0])]

			pos = np.hstack((pos, [1]))
			
			if d == np.nan or d < 0:
				continue
			# if not safe_flag:
			#	 pred = kmeans.predict(np.array([[d]]))
			#	 if pred != largest_cluster_label:
			#		 continue
				
			pos_valid = copy.deepcopy(pos)
			# for j in range(3):
			pos_valid = d * pos

			pos_valid = np.dot(pos_valid, inv_zed_K.T)
			pos_valid = np.dot(pos_valid, R.T) + T.T
			pos_valid = np.dot(pos_valid, mono_K.T)
			pos_valid = pos_valid/pos_valid[:, 2]
			pos_valid = pos_valid.astype(int)
			x_indices = pos_valid[:,1].astype(int)
			y_indices = pos_valid[:,0].astype(int)
			mask = (x_indices >= 0) & (x_indices < zed_img.shape[0]) & (y_indices >= 0) & (y_indices < zed_img.shape[1])
			if not mask:
				continue
			pos_list.append(pos_valid[0])
			index_list.append(i)
			err = math.sqrt(pow(pos_valid[:, 1][0]-int(mono_corner_subpixel[i, 1]), 2) + pow(pos_valid[:, 0][0] - int(mono_corner_subpixel[i, 0]), 2))
			dis = math.sqrt(pow(pos[1]-zed_img.shape[0]/2, 2) + pow(pos[0] - zed_img.shape[1]/2, 2)) 
			err_list.append(err)
			dis_list.append(dis)
			depth_list.append(d/1000)
			sum += err
			td = 1
			for pp in range(10):
				if j < 8:
					print(f"{pp}: {td-0.5} ~ {td}")
					j += 1
				if d/1000 >= td - 0.5 and d/1000 <= td:
					interval_depth_count[pp] += 1
					interval_depth_value[pp] += err
				td += 0.5
		
		mono_rgb = cv2.cvtColor(mono_img, cv2.COLOR_GRAY2BGR)
		zed_rgb = cv2.cvtColor(zed_img, cv2.COLOR_GRAY2BGR)
		
		for i in range(mono_corner_subpixel.shape[0]):
			cv2.circle(mono_rgb, (int(mono_corner_subpixel[i, 0]), int(mono_corner_subpixel[i, 1])), 5, (255, 0, 0), -1)
			# cv2.circle(depth_img, (int(zed_corner_subpixel[i, 0]), int(zed_corner_subpixel[i, 1])), 5, (255, 0, 0), -1)
		for i in range(len(pos_list)):
			cv2.circle(mono_rgb, (pos_list[i][0], pos_list[i][1]), 8, (0, 0, 255))
			# cv2.circle(depth_img, (pos_list[i][0], pos_list[i][1]), 3, (0, 0, 255))
			cv2.circle(depth_img, (int(zed_corner_subpixel[index_list[i], 0]), int(zed_corner_subpixel[index_list[i], 1])), 3, (255, 0, 0), -1)

			
		for i in range(zed_corner_subpixel.shape[0]):
			cv2.circle(zed_rgb, (int(zed_corner_subpixel[i, 0]), int(zed_corner_subpixel[i, 1])), 5, (255, 0, 0), -1)
			
		print(sum/len(pos_list))
		s += sum
		n += len(pos_list)
		cv2.namedWindow("mono", 0)
		cv2.namedWindow("zed", 0)
		cv2.namedWindow("depth_img", 0)
		cv2.imshow("mono", mono_rgb)
		cv2.imshow("zed", zed_rgb)
		cv2.imshow("depth_img", depth_img)
		cv2.waitKey(0)

		# plt.hist(d_roi, bins=20, edgecolor='black', alpha=0.7)
		# plt.xlabel('Value')
		# plt.ylabel('Amount')
		# plt.show()
		# plt.close()

print(f"MAE: {s/n}")
cv2.destroyAllWindows()


plt.scatter(dis_list, err_list, color='blue', marker='o', s=5)
plt.title('Scatter Plot')
plt.xlabel('distance(pixel)')  
plt.ylabel('error (pixel)')
plt.legend()
plt.grid(True)
plt.show()
plt.close()

plt.scatter(depth_list, err_list, color='blue', marker='o', s=5)
plt.title('Scatter Plot')
plt.xlabel('distance(m)')  
plt.ylabel('error (pixel)')
plt.legend()
plt.grid(True)
plt.show()
plt.close()

for i in range(10):
	if interval_depth_count[i] == 0:
		continue
	
	print(f"{i}: {interval_depth_value[i] / interval_depth_count[i]}")