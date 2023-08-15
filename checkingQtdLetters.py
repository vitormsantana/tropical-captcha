import os

# Path to the labeled test set folder
test_folder = r"C:\Users\visantana\Documents\tropical-captcha\labeled_testSet"

# Check for files with incorrect names
incorrect_files = []
for image_filename in os.listdir(test_folder):
    if image_filename.endswith(".png"):
        name_without_extension = image_filename[:-4]  # Remove ".png"
        if len(name_without_extension) != 6:
            incorrect_files.append(image_filename)

# Print incorrect file names
if incorrect_files:
    print("Files with incorrect names:")
    for filename in incorrect_files:
        print(filename)
else:
    print("All files have correct names.")
