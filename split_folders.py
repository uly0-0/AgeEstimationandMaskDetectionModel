import os
import shutil
import random

# Define the source directory containing the images
source_dir = r"C:\Users\ulyss\OneDrive\Desktop\utkcropped"

# Define the destination directories for training and validation sets
train_dir = r"C:\Users\ulyss\OneDrive\Desktop\train"
val_dir = r"C:\Users\ulyss\OneDrive\Desktop\validation"

# Define the desired ratio for splitting the data (e.g., 80% training, 20% validation)
split_ratio = 0.8

# Get a list of all image files in the source directory
image_files = [file for file in os.listdir(source_dir) if file.endswith('.jpg')]

# Sort the image files based on the age information in the filenames
image_files.sort(key=lambda x: int(x.split('_')[0]))

# Calculate the number of images for training and validation sets
num_train = int(len(image_files) * split_ratio)
num_val = len(image_files) - num_train

# Randomly select images for the training and validation sets
train_files = random.sample(image_files, num_train)
val_files = [file for file in image_files if file not in train_files]

# Move or copy the selected images to the respective training and validation folders
for file in train_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.copy(os.path.join(source_dir, file), os.path.join(val_dir, file))
