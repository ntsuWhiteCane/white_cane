import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import curve_fit

start_x, start_y, end_x, end_y = (None, None, None, None)

def select_roi(event, x, y, flags, param):
	global start_x, start_y, end_x, end_y
	if event == cv2.EVENT_LBUTTONDOWN:
		start_x, start_y = x, y
	elif event == cv2.EVENT_LBUTTONUP:
		end_x, end_y = x, y
		cv2.destroyAllWindows()

def normalize_image(data):
	data = data.reshape((360, 640))

	mask = np.isinf(data) | np.isnan(data)

	data[mask] = np.min(data[~mask]) 

	
	data = (data- np.min(data)) / (np.max(data)-np.min(data)) * 255
	canvas = data.astype(np.uint8)
	return canvas

disp_list = []
depth_list = []
for i in range(20, 601, 10):
	ind = str(i)
	disp_path = ".\\fit_data\\Disp\\" + ind + ".npy"
	depth_path = ".\\fit_data\\Depth\\" + ind + ".npy"

	tmp_disp = np.load(disp_path)
	tmp_mask = ~(np.isnan(tmp_disp[:, 0]) | np.isinf(tmp_disp[:, 0]))
	tmp_disp = tmp_disp[tmp_mask]
	if tmp_disp.shape[0] > 500:
		tmp_disp = tmp_disp[::int(tmp_disp.shape[0]/500)]
	disp_list.append(tmp_disp)
	
	tmp_depth = np.load(depth_path)
	tmp_mask = ~(np.isnan(tmp_depth[:, 0]) | np.isinf(tmp_depth[:, 0]))
	tmp_depth = tmp_depth[tmp_mask]
	depth_list.append(tmp_depth)

disp_data = np.vstack(disp_list)
depth_data = np.vstack(depth_list)
depth_data = depth_data[depth_data[:, 1] > 500]
diff = (depth_data[:, 0] - depth_data[:, 1])

boxplot_x = [i/1000 for i in range(1000, 6001, 500)]
depth_box = []
for i in range(1000, 6001, 500):
	depth_box.append(np.abs(diff[depth_data[:, 1] == i])/ i * 100)
# plt.scatter(disp_data[:, 0]/1000, disp_data[:, 1]/1000)
plt.figure()
plt.boxplot(depth_box, positions=boxplot_x, widths=0.2, sym="")
plt.xlabel("Ref(m)", fontsize=20)
plt.ylabel("%error", fontsize=20)
plt.title("Original depth-sensing %error of zed", fontsize=20)
plt.show()
plt.close("all")
print(disp_data.shape)

# # print(np.mean(diff[depth_data[:, 1] == 6000]))

def model_func(disp, a, b):
	return 1 / (a * disp + b)

# Initial fit using RANSAC to identify and remove outliers
# residure threshold is MAD(median absolute deviation)
ransac = RANSACRegressor()
ransac.fit(disp_data[:, 0].reshape(-1, 1), 1 / disp_data[:, 1])
inlier_mask = ransac.inlier_mask_

# Final fitting using only the inliers
popt, pcov = curve_fit(model_func, disp_data[:, 0][inlier_mask]/1000, disp_data[:, 1][inlier_mask]/1000)
# popt, pcov = curve_fit(model_func, disp_data[:, 0]/1000, disp_data[:, 1]/1000)
print("ransac pct:", disp_data[:, 0][inlier_mask].shape[0] / disp_data[:, 0].shape[0])
a, b = popt
print(pcov)
print(f"Estimated parameters: a = {a}, b = {b}")

# Generate fitted curve data
disp_range = np.linspace(min(disp_data[:, 0])/1000, max(disp_data[:, 0])/1000, 500)
fitted_depth = model_func(disp_range, a, b)

# Plot the original data and the fitted curve
plt.figure()
# plt.scatter(disp_data[:, 0]/1000, disp_data[:, 1]/1000, label='Original Data', color='blue')
plt.scatter(disp_data[:, 0][inlier_mask]/1000, disp_data[:, 1][inlier_mask]/1000, color='blue')
plt.scatter(disp_data[:, 0][~inlier_mask]/1000, disp_data[:, 1][~inlier_mask]/1000, color='green')
plt.plot(disp_range, fitted_depth, label=f'Fitted Curve: 1/({a:.4f} * disp + {b:.4f})', color='red')
plt.xlabel('Disparity')
plt.ylabel('Depth')
# plt.title('Fitted Depth vs. Disparity Curve')
plt.title("Disparity-Depth")
plt.legend()
plt.grid(True)
plt.show()
