import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# camera matrix
R = np.array([ [0.999943778871401, -0.007078039364406, 0.007895597199610], [0.006683659654493, 0.998781109540383,   0.048904232116489], [-0.008232119411517, -0.048848711180915,  0.998772264145315]])

T = np.array([ [-56.388641564943384], [-1.755408764448477e+02], [-1.943824622517792e+02]])

zed_K = np.array([[7.059569895868153e+02, 0, 6.263567802746147e+02], [0, 7.065462096249616e+02, 3.765157523308447e+02],[0, 0, 1.0000]])

mono_K = np.array([[6.786368808170881e+02, 0, 6.440856249983478e+02], [0, 6.780005400396101e+02, 3.474849689419397e+02], [0, 0, 1.0000]])


inv_zed_K = np.linalg.inv(zed_K)
inv_R = np.linalg.inv(R)

mono_image_path = ".\\right_ud\\mono35.png"
zed_image_path = ".\\left\\zed_left35.png"
depth_path = ".\\depth\\depth35.npy"

zed_image = cv2.imread(zed_image_path, -1)
mono_image = cv2.imread(mono_image_path, -1)
depth = np.load(depth_path)

# zed_image = zed_image.astype(np.int32) + 50
# zed_image[zed_image[:, :, :] > 255] = 255
# zed_image = zed_image.astype(np.uint8)

# zed_image = cv2.convertScaleAbs(zed_image, alpha=1.4, beta=0)
hsv_image1 = cv2.cvtColor(zed_image, cv2.COLOR_BGR2HSV)
hsv_image2 = cv2.cvtColor(mono_image, cv2.COLOR_BGR2HSV)

# Extract the V channel
h1, s1, v1 = cv2.split(hsv_image1)
h2, s2, v2 = cv2.split(hsv_image2)

# Apply histogram equalization to the V channel
equalized_v1 = cv2.equalizeHist(v1)
equalized_v2 = cv2.equalizeHist(v2)

# Merge the channels back
equalized_hsv_image1 = cv2.merge([h1, s1, equalized_v1])
equalized_hsv_image2 = cv2.merge([h2, s2, equalized_v2])

# Convert back to BGR color space
zed_image = cv2.cvtColor(equalized_hsv_image1, cv2.COLOR_HSV2BGR)
mono_image = cv2.cvtColor(equalized_hsv_image2, cv2.COLOR_HSV2BGR)


cv2.imshow("zed_image", zed_image)
cv2.imshow("mono_image", mono_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

canvas = np.zeros(zed_image.shape, dtype=np.uint8)
pos = 0

print(f"shape of zed image: {zed_image.shape}")

mask = np.zeros((canvas.shape[0], canvas.shape[1]))

# Modify the handling of the concept of unreasonable depth.
x, y = np.meshgrid(np.arange(canvas.shape[1]), np.arange(canvas.shape[0]))
print(f"x.shape: {x.shape}, y.shape: {y.shape}")

pos = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float32)

print(f"1. pos.shape: {pos.shape}")
d = depth[y, x]
valid_depth_indices = np.where((np.isnan(d)) & (d <= 0))

pos_valid = pos.reshape(-1, 3)
pos_valid[valid_depth_indices[0].reshape(-1)] = np.array([0, 0, 1])
d_valid = d.reshape(-1)
d_valid[valid_depth_indices[0].reshape(-1)] = np.array(100000)

print(f"2. pos_valid.shape: {pos_valid.shape}, d_valid.shape: {d_valid.shape}")

# P_h = s[[u], [v], [1]] 
for i in range(3):
     pos_valid[:,i] = d_valid * pos_valid[:,i]

'''
P_w = K^-1 * P_h
because the shape of pos_valid here is n*3, the matrix product need to switch positions

original:
P_w = K^-1 * P_h ((3*1) = (3*3) * (3*1))
To speed up computation, the array needs to be vectorized, so the shape will become n*3
->
Modify
P_w = P_h^T * K^-1^T  ((AB)^T = B^T * A^T)
((n*3) = (n*3) * (3*3))
 ^n positions
'''
pos_valid = np.dot(pos_valid, inv_zed_K.T)

'''
P_new = R*P_w + T
'''
pos_valid = np.dot(pos_valid, R.T) + T.T 

'''
P_new_projection_on_image = K * P_new
'''
pos_valid = np.dot(pos_valid, mono_K.T)

# memory the depth of point
tmp_d = copy.deepcopy(pos_valid[:, 2])
tmp_d = tmp_d.astype(int)

# mask of negative depth
neg_depth_mask = pos_valid[:, 2] <= 0

neg_depth_mask = neg_depth_mask.reshape(zed_image.shape[0], zed_image.shape[1])

'''
normalize:
P_new_projection_on_image = [[u_x * z], [u_y * z], [z]] = z[[u_x], [u_y], [1]]
'''
for i in range(3):
     pos_valid[:,i] = pos_valid[:,i]/pos_valid[:,2]
print(f"3. pos_valid.shape: {pos_valid.shape}")

pos_valid = pos_valid.astype(int)

'''
Because the projection will reduce the dimension, there may be some point overlap. The desired pixel is the nearest point.
unique method will return

unique_elements: All elements of the array, but each repeated element will appear only once.
first_indices: The indeices
first_indeices: The index corresponds to the position of the first occurrence of each unique row in the original array.
counts: The amount of each unique element. 
'''
unique_elements, first_indices, counts = np.unique(pos_valid, axis=0, return_index=True, return_counts=True)

# get the position mask of elements that repeat in the array
tmp_mask = ~(counts == 1)
# get the elements that repeat in the array
tmp_first_indices = first_indices[tmp_mask]

tmp_pos_valid = copy.deepcopy(pos_valid)
tmp_pos_valid[:, 2] = tmp_d

# Find the nearest point among the overlapping points in the array.
target_set = {tuple(row) for row in pos_valid[tmp_first_indices]}
logical_array = np.array([tuple(row) in target_set for row in pos_valid])
pos_notValid = tmp_pos_valid[logical_array]
print(f"pos_notValid.shape: {pos_notValid.shape}")

logical_array = ~logical_array

mem = pos_notValid[0]
order = np.argsort(pos_notValid[:, 2])
pos_notValid = pos_notValid[order]

_, unique_indices = np.unique(pos_notValid[:, :2], axis=0, return_index=True)
pos_notValid_pass = pos_notValid[unique_indices]
origin_order = order[unique_indices]

logical_array[origin_order] = True
logical_array = logical_array.reshape(zed_image.shape[0], zed_image.shape[1])

pos_valid = pos_valid.reshape(zed_image.shape[0], zed_image.shape[1], 3)
print(f"4. pos_valid.shape: {pos_valid.shape}")

x_indices = pos_valid[:,:,1].astype(int)
y_indices = pos_valid[:,:,0].astype(int)

# get the mask of the points that are out of the range.
mask = (x_indices >= 0) & (x_indices < canvas.shape[0]) & (y_indices >= 0) & (y_indices < canvas.shape[1])

# the depth measurement at the edge of object is really bad so mask it.
edgeimg = cv2.Canny(zed_image, 50, 200)
dilate = cv2.dilate(edgeimg, kernel=np.ones((3,3)))
edge_mask = dilate == 0
mask = edge_mask & mask
mask = mask & logical_array
mask = mask & ~neg_depth_mask

# put the mask on image
canvas[x_indices[mask], y_indices[mask]] = zed_image[mask].astype(np.uint8)
point_projection_on_image = np.zeros((zed_image.shape[0], zed_image.shape[1])).astype(np.uint8)
point_projection_on_image[x_indices[mask], y_indices[mask]] = 255
mask = (point_projection_on_image == 255)

canvas2 = np.zeros(mono_image.shape, dtype=np.uint8)
tmpimg = mono_image.copy()
canvas2[mask] = tmpimg[mask]

cv2.imshow("mask", mask.astype(np.uint8)*255)
diff = cv2.absdiff(cv2.cvtColor(canvas2, cv2.COLOR_BGR2GRAY), cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY))  
diff[~mask] = 0

cv2.imshow("mono that is masked", canvas2)
cv2.imshow("projection image", canvas)
cv2.imshow("diff", diff)
cv2.imshow("zed", zed_image)
cv2.imshow("mono", mono_image)
cv2.imshow("point projection result", point_projection_on_image)
cv2.waitKey(0)

gray_image1 = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(canvas2, cv2.COLOR_BGR2GRAY)

# compute MAE
allerror = np.sum(diff[mask])
psnr = cv2.PSNR(gray_image1, gray_image2)

print("mean error: ", allerror/np.count_nonzero(mask))
print("count: ", np.count_nonzero(mask))
print("psnr: ", psnr, "db")


from skimage.metrics import structural_similarity as ssim

score, diff = ssim(gray_image1, gray_image2, full=True, win_size=11)

print(f"SSIM: {score}")

# display diff image 
diff = (diff * 255).astype("uint8")
cv2.imshow("Difference", diff)

cv2.waitKey(0)
cv2.destroyAllWindows()


