import os
import re
import shutil

# Define the age groups
ageList = ['(0-3)', '(4-7)', '(8-14)', '(15-20)', '(21-32)', '(33-43)', '(44-53)', '(54-100)']

# Define the function to sort files into folders
def sort_files_by_age(file_dir):
    # Create subfolders for each age group if they don't exist
    for age_group in ageList:
        if not os.path.exists(os.path.join(file_dir, age_group)):
            os.makedirs(os.path.join(file_dir, age_group))

    # Get a list of files in the directory
    files = os.listdir(file_dir)

    # Iterate through the files
    for file_name in files:
        # Extract the first number using regular expressions
        match = re.match(r'^(\d+)_.*', file_name)
        if match:
            first_number = int(match.group(1))
            # Find the appropriate age group
            for i in range(len(ageList)):
                age_range = re.findall(r'\d+', ageList[i])  # Extract lower and upper bounds of age range
                lower_bound = int(age_range[0])
                upper_bound = int(age_range[1])
                if lower_bound <= first_number <= upper_bound:
                    # Move the file to the corresponding age group folder using shutil.move
                    old_path = os.path.join(file_dir, file_name)
                    new_path = os.path.join(file_dir, ageList[i], file_name)
                    shutil.move(old_path, new_path)
                    break

# Specify the directory containing the files
file_directory = r"C:\Users\ulyss\OneDrive\Desktop\dataset\validation"

# Call the function to sort files into folders
sort_files_by_age(file_directory)
