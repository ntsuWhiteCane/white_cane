import numpy as np

lidar1_R = np.load(".\\data_list\\lidar1\\R.npy")
lidar1_T = np.load(".\\data_list\\lidar1\\T.npy")

zed1_R = np.load(".\\data_list\\zed1\\R.npy")
zed1_T = np.load(".\\data_list\\zed1\\T.npy")

R_1 = np.dot(zed1_R, lidar1_R)
T_1 = np.dot(zed1_R, lidar1_T) + zed1_T 

print(R_1)
print(T_1)

lidar2_R = np.load(".\\data_list\\lidar2\\R.npy")
lidar2_T = np.load(".\\data_list\\lidar2\\T.npy")

zed2_R = np.load(".\\data_list\\zed2\\R.npy")
zed2_T = np.load(".\\data_list\\zed2\\T.npy")
R_2 = np.dot(zed2_R, lidar2_R)
T_2 = np.dot(zed2_R, lidar2_T) + zed2_T 

print(R_2)
print(T_2)

R_3 = np.dot(R_1.T, R_2)
T_3 = np.dot(R_1.T, T_2) - T_1

print(R_3)
print(T_3)