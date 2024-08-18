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
        
        self.__four_points = []
        self.__four_count = 0
        self.__four_flag = True
    def select_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_x, self.end_y = x, y
            cv2.destroyAllWindows()
            self.detect = True
            
    def __select_4_point(self, event, x, y, flags, param):
        # self.__four_count = 0 
        if event == cv2.EVENT_LBUTTONDOWN and self.__four_flag:
            self.__four_points.append([x, y])
            self.__four_flag = False
        elif event == cv2.EVENT_LBUTTONUP:
            self.__four_count += 1
            self.__four_flag = True 
        
        if self.__four_count >= 4:
            cv2.destroyAllWindows()
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
        print(f"start_x: {self.start_x}, start_y: {self.start_y}, end_x: {self.end_x}, end_y: {self.end_y}")
        if self.start_x is None:
            return
        min_x = self.start_x - 30 + canvas_col_min
        max_x = self.end_x - 30 + canvas_col_min
        min_z = -1 * (self.start_y - 30 + canvas_row_min)
        max_z = -1 * (self.end_y - 30 + canvas_row_min)

        self.x_roi = np.array([min_x / 100, max_x / 100])
        self.z_roi = np.array([max_z / 100, min_z / 100])
    def select_4_points(self):
        canvas_col_min = np.min(self.data[:, 0])
        canvas_row_min = np.min(self.data[:, 1])
        row = int(np.max(self.data[:, 1]) - canvas_row_min) + 60
        col = int(np.max(self.data[:, 0]) - canvas_col_min) + 60
        self.canvas = np.zeros((row, col, 3), dtype=np.uint8)
        for i in range(self.data.shape[0]):
            x = int(self.data[i, 0] - canvas_col_min) + 30
            y = int(self.data[i, 1] - canvas_row_min) + 30
            cv2.circle(self.canvas, (x, y), 1, (255, 0, 0))
        cv2.destroyAllWindows()
        cv2.namedWindow("select lidar 4 point", 0)
        cv2.setMouseCallback('select lidar 4 point', self.__select_4_point)
        cv2.imshow("select lidar 4 point", self.canvas)
        cv2.waitKey(0)

        four_point = np.array(self.__four_points)
        
        lidar_four_point = []
        
        for i in range(4):
            x = four_point[i, 0] - 30 + canvas_col_min
            y = four_point[i, 1] - 30 + canvas_row_min 
            tmp = np.linalg.norm(self.data-np.array([x, y]), axis=-1)
            tmp_min = np.argmin(tmp)
            lidar_four_point.append(self.data[tmp_min])
        lidar_four_point = np.array(lidar_four_point)
        for i in range(4):
            x = int(lidar_four_point[i, 0] - canvas_col_min) + 30
            y = int(lidar_four_point[i, 1] - canvas_row_min) + 30
            cv2.circle(self.canvas, (x, y), 1, (0, 255, 0))
        # cv2.imshow("4 points", self.canvas)
        # cv2.waitKey(0) 
        cv2.destroyAllWindows()
        return lidar_four_point / 100
        

