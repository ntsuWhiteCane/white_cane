import cv2
import copy
import math
import numpy as np

class CameraRoi():
    def __init__(self, img:np.ndarray, depth:np.ndarray, intrinsic_K:np.ndarray):
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.detect = False
        self.data = img
        self.depth = depth
        self.canvas = None
        self.K = intrinsic_K
        self.inv_K = np.linalg.inv(self.K)
    def select_roi(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_x, self.start_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_x, self.end_y = x, y
            cv2.destroyAllWindows()
            self.detect = True
    def show_figure(self):
        
        self.canvas = self.data

        cv2.namedWindow("select camera roi")
        cv2.setMouseCallback('select camera roi', self.select_roi)
        cv2.imshow("select camera roi", self.canvas)
        cv2.waitKey(0)
        print(f"start_x: {self.start_x}, start_y: {self.start_y}, end_x: {self.end_x}, end_y: {self.end_y}")
        # cv2.imshow("qq", self.data[self.start_y:self.end_y, self.start_x:self.end_x])

    def computePoint(self) -> np.ndarray | None:
        self.canvas = np.zeros((self.data.shape[0], self.data.shape[1])) 
        self.canvas = self.canvas.astype(np.float32)
        if (self.start_x is None) or (self.end_x is None) or (self.start_y is None) or (self.end_y is None) or (self.end_x <= self.start_x) or (self.end_y <= self.start_y):
            print("plz select proper roi")
            return None
        pos = self.depth.reshape((-1, 3))
        pos = np.dot(pos, self.K.T)
        for i in range(2):
            pos[:,i] = pos[:,i] / pos[:, 2]

        tmp_mask = pos[:, 0] >= 0
        tmp_mask &= pos[:, 0] < self.canvas.shape[1]
        tmp_mask &= pos[:, 1] >= 0
        tmp_mask &= pos[:, 1] < self.canvas.shape[0]
        
        valid = pos[tmp_mask].astype(np.int32)
        self.canvas[valid[:, 1], valid[:, 0]] = valid[:, 2]

        tmp_mask1 = pos[:, 0] > self.start_x
        tmp_mask1 &= pos[:, 0] < self.end_x
        tmp_mask1 &= pos[:, 1] > self.start_y
        tmp_mask1 &= pos[:, 1] < self.end_y
        
        valid_pos = pos[tmp_mask1]
        
        # valid_pos = valid_pos.reshape((-1, 3))
        for i in range(2):
            valid_pos[:,i] = valid_pos[:, 2] * valid_pos[:,i]
        valid_pos = np.dot(valid_pos, self.inv_K.T) 
        return valid_pos / 1000


