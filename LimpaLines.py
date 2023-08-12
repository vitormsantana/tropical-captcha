# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 17:07:01 2023

@author: visantana
"""

import os
import time 

#Filtrar os que n tenha 4lines OU 4_lines e deletar o restante

folder_path = r"C:\Users\visantana\Documents\tropical-captcha\Divided_Captchas"

try:
    for filename in os.listdir(folder_path):
        if "4_lines" not in filename and "4lines" not in filename:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
    
    print("Deletion completed.")
except Exception as e:
    print(f"An error occurred: {e}")
    print(f"An error occurred: {e}")
   

folder_path = r"C:\Users\visantana\Documents\tropical-captcha\Divided_Captchas"


# Iterate through the files in the directory
for filename in os.listdir(folder_path):
    if "_" in filename:
        new_filename = filename.replace("_", "")
        timestamp = int(time.time())  # Get current timestamp
        unique_new_filename = f"{new_filename}"
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, unique_new_filename)
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {unique_new_filename}")
    

folder_path = r"C:\Users\visantana\Documents\tropical-captcha\Divided_Captchas"

try:
    files_to_delete = []
    filenames = sorted(os.listdir(folder_path), key=lambda x: x.lower())  # Sort ignoring case
    prev_prefix = ""
    prev_file = ""
    
    for filename in filenames:
        prefix = filename[:14]
        if prefix == prev_prefix and prev_file != "":
            files_to_delete.append(prev_file)
        prev_prefix = prefix
        prev_file = filename
    
    for file_to_delete in files_to_delete:
        os.remove(os.path.join(folder_path, file_to_delete))
        print(f"Deleted file: {file_to_delete}")
    
    print("Deletion completed.")
except Exception as e:
    print(f"An error occurred: {e}")
        