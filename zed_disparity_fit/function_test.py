import matplotlib.pyplot as plt
import numpy as np

# a = np.load("data\\Imu\\0001.npy")
# print(a)

# yaw_list = []
# for i in range(1, 1500):
# 	data_path = ".\\data\\Imu\\" + str(i).zfill(4) + ".npy"
# 	imu = np.load(data_path)
# 	yaw_list.append([imu[0], i])

# yaw_list = np.array(yaw_list)
# plt.plot(yaw_list[1000:1400, 1], np.cos(yaw_list[1000:1400, 0]))
# plt.xlabel("time(ticks)")
# plt.ylabel("cos(theta)")
# plt.title("mpu6050 dynamic performance")
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.boxplot(data, labels=['Group 1', 'Group 2', 'Group 3'])
plt.title('Box Plot using Matplotlib')
plt.xlabel('Groups')
plt.ylabel('Values')
plt.show()
