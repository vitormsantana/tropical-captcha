import os
from collections import defaultdict

# Specify the directory path
directory_path = r"C:\Users\visantana\Documents\tropical-captcha\Divided_Captchas"

# Get a list of all files in the directory
all_files = os.listdir(directory_path)

# Create a dictionary to group files by their last 12 digits
file_groups = defaultdict(list)
for file_name in all_files:
    last_12_digits = file_name[-12:]
    file_groups[last_12_digits].append(file_name)

# Delete duplicate files
for group in file_groups.values():
    if len(group) > 1:
        for file_name in group[1:]:
            file_path = os.path.join(directory_path, file_name)
            os.remove(file_path)
            print(f"Deleted: {file_path}")