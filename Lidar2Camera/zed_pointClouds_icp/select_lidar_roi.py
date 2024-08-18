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
    def select_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_x, self.end_y = x, y
            cv2.destroyAllWindows()
            self.detect = True
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
        cv2.namedWindow("select lidar roi")
        cv2.setMouseCallback('select lidar roi', self.select_roi)
        cv2.imshow("select lidar roi", self.canvas)
        cv2.waitKey(0)
        print(f"start_x: {self.start_x}, start_y: {self.start_y}, end_x: {self.end_x}, end_y: {self.end_y}")
        min_x = self.start_x - 30 + canvas_col_min
        max_x = self.end_x - 30 + canvas_col_min
        min_z = -1 * (self.start_y - 30 + canvas_row_min)
        max_z = -1 * (self.end_y - 30 + canvas_row_min)

        self.x_roi = np.array([min_x / 100, max_x / 100])
        self.z_roi = np.array([max_z / 100, min_z / 100])
        

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

# data_index = "0001"
# lidar_point_cloud_path = "PointClouds\\" + data_index + ".pcd"

# points = read_pcd(lidar_point_cloud_path)
# points = np.stack((-1*points[:, 1], -1*points[:, 2], points[:, 0]), axis=-1)
# dd = LidarRoi(points)
# dd.show_figure()
# print(dd.x_roi, dd.z_roi)
