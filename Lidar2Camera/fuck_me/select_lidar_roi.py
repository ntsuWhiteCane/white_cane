import math

import cv2
import numpy as np

class LidarRoi():
	def __init__(self, lidar_point:np.ndarray):
		self.start_x = None
		self.start_y = None
		self.end_x = None
		self.end_y = None
		self.x_roi = np.zeros(2)
		self.z_roi = np.zeros(2)
		self.detect = False
		self.data = np.stack((lidar_point[:, 0], -1*lidar_point[:, 2]), axis=-1)*100
		self.canvas = None
		
		self.__n = 0
		self.__n_count = 0
		self.__n_flag = True
		self.__n_points = []
	def select_roi(self, event, x, y, flags, param):
		if event == cv2.EVENT_LBUTTONDOWN:
			self.start_x, self.start_y = x, y

		elif event == cv2.EVENT_LBUTTONUP:
			self.end_x, self.end_y = x, y
			cv2.destroyAllWindows()
			self.detect = True


	def __select_n_point(self, event, x, y, flags, param):
		
		if event == cv2.EVENT_LBUTTONDOWN and self.__n_flag:
			self.__n_points.append([x, y])
			self.__n_flag = False
		elif event == cv2.EVENT_LBUTTONUP:
			self.__n += 1
			self.__n_flag = True 
		
		if self.__n_count >= self.__n:
			cv2.destroyAllWindows()

	def create_canvas(self):
		canvas_col_min = np.min(self.data[:, 0])
		canvas_row_min = np.min(self.data[:, 1])
		row = int(np.max(self.data[:, 1]) - canvas_row_min) + 60
		col = int(np.max(self.data[:, 0]) - canvas_col_min) + 60
		self.canvas = np.zeros((row, col, 3), dtype=np.uint8)

		for i in range(self.data.shape[0]):
			x = int(self.data[i, 0] - canvas_col_min) + 30
			y = int(self.data[i, 1] - canvas_row_min) + 30
			cv2.circle(self.canvas, (x, y), 1, (255, 0, 0))
		
	def show_figure(self):
		canvas_col_min = np.min(self.data[:, 0])
		canvas_row_min = np.min(self.data[:, 1])
		row = int(np.max(self.data[:, 1]) - canvas_row_min) + 60
		col = int(np.max(self.data[:, 0]) - canvas_col_min) + 60
		self.canvas = np.zeros((row, col, 3), dtype=np.uint8)

		for i in range(self.data.shape[0]):
			x = int(self.data[i, 0] - canvas_col_min) + 30
			y = int(self.data[i, 1] - canvas_row_min) + 30
			cv2.circle(self.canvas, (x, y), 1, (255, 0, 0))
		cv2.namedWindow("select lidar roi", 0)
		cv2.setMouseCallback('select lidar roi', self.select_roi)
		cv2.imshow("select lidar roi", self.canvas)
		cv2.waitKey(0)
		# print(f"start_x: {self.start_x}, start_y: {self.start_y}, end_x: {self.end_x}, end_y: {self.end_y}")
		if self.start_x is None:
			return
		min_x = self.start_x - 30 + canvas_col_min
		max_x = self.end_x - 30 + canvas_col_min
		min_z = -1 * (self.start_y - 30 + canvas_row_min)
		max_z = -1 * (self.end_y - 30 + canvas_row_min)

		self.x_roi = np.array([min_x / 100, max_x / 100])
		self.z_roi = np.array([max_z / 100, min_z / 100])

	def select_n_points(self, n = 3):
		canvas_col_min = np.min(self.data[:, 0])
		canvas_row_min = np.min(self.data[:, 1])
		row = int(np.max(self.data[:, 1]) - canvas_row_min) + 60
		col = int(np.max(self.data[:, 0]) - canvas_col_min) + 60
		self.canvas = np.zeros((row, col, 3), dtype=np.uint8)
		self.__n = n

		for i in range(self.data.shape[0]):
			x = int(self.data[i, 0] - canvas_col_min) + 30
			y = int(self.data[i, 1] - canvas_row_min) + 30
			cv2.circle(self.canvas, (x, y), 1, (255, 0, 0))
		cv2.destroyAllWindows()
		cv2.namedWindow(f"select lidar {n} point", 0)
		cv2.setMouseCallback(f"select lidar {n}", self.__select_n_point)
		cv2.imshow(f"select lidar {n} point", self.canvas)
		cv2.waitKey(0)

		n_point = np.array(self.__n_points)
		
		lidar_n_point = []
		
		for i in range(3):
			x = n_point[i, 0] - 30 + canvas_col_min
			y = n_point[i, 1] - 30 + canvas_row_min 
			tmp = np.linalg.norm(self.data-np.array([x, y]), axis=-1)
			tmp_min = np.argmin(tmp)
			lidar_n_point.append(self.data[tmp_min])
		lidar_n_point = np.array(lidar_n_point)
		for i in range(3):
			x = int(lidar_n_point[i, 0] - canvas_col_min) + 30
			y = int(lidar_n_point[i, 1] - canvas_row_min) + 30
			cv2.circle(self.canvas, (x, y), 1, (0, 255, 0))
		# cv2.namedWindow("3 points", 0)
		# cv2.imshow("3 points", self.canvas)
		# cv2.waitKey(0) 
		cv2.destroyAllWindows()
		lidar_n_point[:, 1] = -1*lidar_n_point[:, 1]
		return lidar_n_point / 100
	
	def show_points(self, point, roi=(0, 0, 0, 0), roi_color=(0, 0, 255)):
		points = point*100
		points[:, 2] = -1*points[:, 2]
		points = np.stack((point[:, 0], -1*point[:, 2]), axis=-1)*100

		canvas_col_min = np.min(points[:, 0])
		canvas_row_min = np.min(points[:, 1])
		row = int(np.max(points[:, 1]) - canvas_row_min) + 60
		col = int(np.max(points[:, 0]) - canvas_col_min) + 60
		self.canvas = np.zeros((row, col, 3), dtype=np.uint8)
		# print(row, col, canvas_col_min, canvas_row_min, np.max(points[:, 0]), np.max(points[:, 1]), row, col)

		for i in range(points.shape[0]):
			x = int(points[i, 0] - canvas_col_min) + 30
			y = int(points[i, 1] - canvas_row_min) + 30
			if point[i, 0] < roi[0] or point[i, 0] > roi[1] or point[i, 2] < roi[2] or point[i, 2] > roi[3]:
				cv2.circle(self.canvas, (x, y), 1, (255, 0, 0))
			else:
				cv2.circle(self.canvas, (x, y), 1, roi_color)
		cv2.destroyAllWindows()
		cv2.namedWindow("points", 0)	
		cv2.imshow("points", self.canvas)
		key = cv2.waitKey(0)	
		return key