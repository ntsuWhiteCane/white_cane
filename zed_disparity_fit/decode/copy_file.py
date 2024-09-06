import os
import shutil

# File paths
data_file_path = 'record.txt'  # Path to the text file containing image numbers
source_folder = "E:\\white_cane\\decode_rosbag_data\\data\\Images"         # Source folder where the images are stored
destination_folder = ".\\data\\Images"     # Destination folder where the images will be copied to

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Read all numbers from the data file
with open(data_file_path, 'r') as file:
    numbers = file.read().splitlines()

# Iterate through each number
for number in numbers:
    # Construct the source file path
    src_file = os.path.join(source_folder, f'{number}.png')
    
    # Construct the destination file path
    dst_file = os.path.join(destination_folder, f'{number}.png')
    
    # Check if the source file exists
    if os.path.isfile(src_file):
        # Copy the file to the destination folder
        shutil.copy(src_file, dst_file)
        print(f'Copied {src_file} to {dst_file}')
    else:
        print(f'Source file {src_file} does not exist')

print('All copy operations completed')
