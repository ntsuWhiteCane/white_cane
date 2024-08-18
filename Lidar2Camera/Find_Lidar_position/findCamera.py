import cv2
import numpy as np

block_size = 80

zed_K = np.array([[3.506789914489589e+02, 0, 3.127677577917245e+02],
                  [0, 3.510410495486862e+02, 1.875433445883912e+02],
                  [0, 0, 1]])

ind = 6
data_index = str(ind).zfill(4)
image_path = ".\\find_position\\Images\\0006.png"

image_points = []
image = cv2.imread(image_path, -1)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(image_gray.shape)
p = None
def select_roi(event, x, y, flags, param):
    global image_points
    global image_gray
    global image
    
    global p
    if event == cv2.EVENT_LBUTTONDOWN:
        p = np.array([[x, y]], dtype=np.float32)
    if event == cv2.EVENT_LBUTTONUP:
        print(p)
        

        winSize = (10, 10)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 40, 0.001)
        point = cv2.cornerSubPix(image_gray, p, winSize, zeroZone, criteria)
        # cv2.destroyWindow("hihi")
        cv2.circle(image, (int(point[0, 0]), int(point[0, 1])), 4, (255, 0, 0), -1)
        cv2.imshow("hihi", image)

        image_points.append(point[0])

    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.destroyAllWindows()

cv2.namedWindow("hi")
cv2.imshow("hi", image_gray)
cv2.imshow("hihi", image)
cv2.setMouseCallback('hi', select_roi)
cv2.waitKey(0)

image_points = np.array(image_points, dtype=np.float32)
print(image_points)

x = int(input("x axis point: "))
y = int(input("y axis point: "))

world_point = []

for j in range(y):
    for i in range(x):
        world_point.append([block_size + i*block_size, block_size + j*block_size, 0])
print(world_point)

world_points = np.array(world_point, dtype=np.float32)

distCoeffs = np.zeros((5,1))

success, rvecs, tvecs = cv2.solvePnP(world_points, image_points, zed_K, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)

np_rodrigues = np.asarray(rvecs[:,:],np.float64)
rmat = cv2.Rodrigues(np_rodrigues)[0]
camera_position = -np.matrix(rmat).T * np.matrix(tvecs)

print(rmat)
print(camera_position)
print(tvecs)

camera_position = camera_position/1000