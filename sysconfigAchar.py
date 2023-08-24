import os

# Path to the site-packages directory of the Conda environment
conda_env_path = r'C:\Users\visantana\AppData\Local\anaconda3\envs\my_tf_env\Lib\site-packages'

def list_folders_and_subfiles(directory):
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            print(f"Folder: {os.path.join(root, folder)}")
        for file in files:
            print(f"File: {os.path.join(root, file)}")

list_folders_and_subfiles(conda_env_path)
