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
for i in range(30, 601, 10):
	ind = str(i)
	disp_path = ".\\test_data\\Disp\\" + ind + ".npy"
	depth_path = ".\\test_data\\Depth\\" + ind + ".npy"

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
diff = (depth_data[:, 0] - depth_data[:, 1])
diff_mean = []
plot_x = [i/1000 for i in range(500, 6001, 100)]
for i in range(500, 6001, 100):
	# print(f"{i}.", np.mean(np.abs(diff[disp_data[:, 1] == i])))
	diff_mean.append(np.mean(np.abs(diff[depth_data[:, 1] == i])) /(i) * 100)
	# plt.scatter(i/1000, np.mean(np.abs(diff[disp_data[:, 1] == i]) )/(i / 1000) * 100, c='b')
plt.plot(plot_x, diff_mean)
plt.xlabel("Ref(m)", fontsize=20)
plt.ylabel("%error", fontsize=20)
plt.yticks([i for i in range(1, 10)])
# plt.xlim((0, 7))
# plt.ylim((0, 16))
plt.title("Original depth-sensing %error of zed", fontsize=20)
plt.show()
plt.close("all")

print(np.mean(np.abs(diff/depth_data[:, 1] * 100)))
# print(disp_data.shape)

# print(np.mean(diff[depth_data[:, 1] == 6000]), "mm")

def model_func(disp, a=-23.729979928848525, b=-0.014739943832346883):
    return 1 / (a * disp + b)
a=-23.730672667249834
b= -0.0147465843007603095

# Generate fitted curve data
disp_range = np.linspace(min(disp_data[:, 0])/1000, max(disp_data[:, 0])/1000, 500)
fitted_depth = model_func(disp_range, a, b)

# Plot the original data and the fitted curve
# plt.figure()
# plt.scatter(disp_data[:, 0]/1000, disp_data[:, 1]/1000, color='blue')
# plt.plot(disp_range, fitted_depth, color='red')
# plt.xlabel('Disparity')
# plt.ylabel('Depth')
# plt.title('Fitted Depth vs. Disparity Curve')
# # plt.legend()
# plt.grid(True)
# plt.show()
# plt.close()
diff = disp_data[:, 1]/1000 - model_func(disp_data[:, 0]/1000, a, b)
# for i in range(500, 6001, 100):
	# print(f"{i}.", np.mean(diff[disp_data[:, 1] == i]))

# print(au[disp_data[:, 1] == 200])
boxplot_x = [i/1000 for i in range(1000, 6001, 500)]
depth_box = []
for i in range(200, 6001, 100):
	# print(f"{i}.", np.mean(np.abs(diff[disp_data[:, 1] == i])))
	diff_mean.append(np.mean(np.abs(diff[disp_data[:, 1] == i])) /(i / 1000) * 100)
for i in range(1000, 6001, 500):
	depth_box.append(np.abs(diff[disp_data[:, 1] == i])/ (i / 1000) * 100)
# plt.scatter(disp_data[:, 0]/1000, disp_data[:, 1]/1000)
plt.figure()
plt.boxplot(depth_box, positions=boxplot_x, widths=0.2, sym="")
# plt.plot(plot_x, diff_mean)
plt.xlabel("Ref(m)", fontsize=20)
plt.ylabel("%error", fontsize=20)
# plt.xlim((0, 7))
# plt.ylim((0, 16))
plt.title("depth-sensing %error of zed(RANSAC)", fontsize=20)
plt.show()
plt.close("all")
diff_mean = []
plot_x = [i/1000 for i in range(200, 6001, 100)]
for i in range(200, 6001, 100):
	# print(f"{i}.", np.mean(np.abs(diff[disp_data[:, 1] == i])))
	diff_mean.append(np.mean(np.abs(diff[disp_data[:, 1] == i])) /(i / 1000) * 100)
	# plt.scatter(i/1000, np.mean(np.abs(diff[disp_data[:, 1] == i]) )/(i / 1000) * 100, c='b')
plt.plot(plot_x, diff_mean)
plt.xlabel("Ref(m)", fontsize=20)
plt.ylabel("%error", fontsize=20)
# plt.xlim((0, 7))
# plt.ylim((0, 16))
plt.title("depth-sensing %error of zed", fontsize=20)
plt.show()
plt.close("all")
# print(np.mean(diff_mean))