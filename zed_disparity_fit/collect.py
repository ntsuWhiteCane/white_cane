import cv2
import numpy as np

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

for i in range(20, 601, 10):
	ind = str(i)
	img_path = ".\\test\\Images\\" + ind + ".png"
	disp_path = ".\\test\\Disp\\" + ind + ".npy"
	depth_path = ".\\test\\Depth\\" + ind + ".npy"
	output_disp_path = "test_data\\Disp\\" + ind + ".npy"
	output_depth_path = "test_data\\Depth\\" + ind + ".npy"

	img = cv2.imread(img_path, -1)
	disp = np.load(disp_path).reshape((360, 640))
	depth = np.load(depth_path).reshape((360, 640))
	disp_img = normalize_image(disp)
	cv2.namedWindow("img", 0)
	cv2.setMouseCallback("img", select_roi)

	cv2.imshow("disp", disp_img)
	cv2.imshow("img", img)
	cv2.waitKey(0)

	img_roi = img[start_y:end_y, start_x:end_x]
	depth_roi = depth[start_y:end_y, start_x:end_x].reshape(-1)
	disp_roi = disp[start_y:end_y, start_x:end_x].reshape(-1)

	cv2.imshow("img_roi", img_roi)
	key = cv2.waitKey(0)
	cv2.destroyAllWindows()

	disp_label = np.zeros((disp_roi.shape[0], 2), dtype=np.float32)
	depth_label = np.zeros((depth_roi.shape[0], 2), dtype=np.float32)
	# print(np.min(disp_roi[~(np.isinf(disp_roi) | np.isnan(disp_roi))]))
	disp_label[:, 0] = disp_roi
	disp_label[:, 1] = i * 10 # unit: mm
	depth_label[:, 0] = depth_roi*1000
	depth_label[:, 1] = i * 10

	if key == 27:
		break

	if key != 'n':
		print(i)
		np.save(output_depth_path, depth_label)
		np.save(output_disp_path, disp_label)