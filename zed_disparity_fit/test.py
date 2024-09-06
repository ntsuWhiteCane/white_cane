import cv2
import numpy as np

a = np.load(".\\data\\Disp\\100.npy")
print(a.shape)
a = a.reshape((360, 640))

mask = np.isinf(a) | np.isnan(a)

a[mask] = np.min(a[~mask]) 

print(np.min(a))
print(np.max(a))
a = (a- np.min(a)) / (np.max(a)-np.min(a)) * 255
a = a.astype(np.uint8)

cv2.imshow("test", a)
cv2.waitKey(0)


