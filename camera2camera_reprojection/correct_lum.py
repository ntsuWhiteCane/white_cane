import cv2
import numpy as np

# Read the images
image1 = cv2.imread('left\\zed_left1.png')
image2 = cv2.imread('right_ud\\mono1.png')

# Convert images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute grayscale histograms
hist1 = cv2.calcHist([gray_image1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([gray_image2], [0], None, [256], [0, 256])

# Apply histogram equalization
equalized_image1 = cv2.equalizeHist(gray_image1)
equalized_image2 = cv2.equalizeHist(gray_image2)

# Convert to HSV color space
hsv_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
hsv_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

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
equalized_image1 = cv2.cvtColor(equalized_hsv_image1, cv2.COLOR_HSV2BGR)
equalized_image2 = cv2.cvtColor(equalized_hsv_image2, cv2.COLOR_HSV2BGR)



cv2.imshow("image1", equalized_image1)
cv2.imshow("image2", equalized_image2)
cv2.waitKey(0)