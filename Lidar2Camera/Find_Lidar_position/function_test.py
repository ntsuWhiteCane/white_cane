import numpy as np

v1 = np.array([1, 2, 3, 0])
v2 = np.array([4, 5, 6, 2])
v3 = np.array([7, 8, 9, 1])
# print(v1.reshape(3, -1))


a = np.stack((v1, v2, v3))
print(a)

b = v1 + v2 + v3 
print(b)
print(b/4)

print(np.linalg.norm(a, axis=-1))
print(v1.reshape((4, 1)).shape)
print(v1.shape)
v1 = np.matrix(v1)
print(v1)
print(v1.shape)

dd = np.array([[1, 2], [3, 4]])
dd2 = np.array([[3,4], [5,6], [7, 8]])

print(np.vstack((dd, dd2)))

a = np.array([[1], [2], [3]])
print(a)
print(a.T)