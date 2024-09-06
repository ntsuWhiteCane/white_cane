import cv2
import os

# Define the path to the directory containing the images
image_folder = 'test_img\\'

# Create a window to display the images
cv2.namedWindow('Image Sequence', cv2.WINDOW_NORMAL)
# img = cv2.imread(image_folder + "0001.png")
# cv2.imshow("i", img)
# cv2.waitKey(0)
# Loop through each image in the sequence
for i in range(1, 223):  # 0000.png to 0300.png, so 301 images in total
    # Generate the filename with leading zeros
    filename = os.path.join(image_folder, f'{i:04d}.png')
    
    # Read the image
    img = cv2.imread(filename)
    
    # If the image was successfully read
    if img is not None:
        # Display the image in the window
        cv2.imshow('Image Sequence', img)
        
        # Wait for a short duration before showing the next image (e.g., 30ms)
        if cv2.waitKey(300) & 0xFF == ord('q'):
            break
    else:
        print(f'Image {filename} not found or unable to load.')
        break

# Release the window
cv2.destroyAllWindows()
