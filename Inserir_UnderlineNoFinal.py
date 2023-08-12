# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:04:49 2023

@author: visantana
"""

import os
folder_path = r"C:\Users\visantana\Documents\tropical-captcha\Divided_Captchas"
#Inserir _ ao final de todos os nomes
    
try:
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".png"):
            new_filename = filename.rsplit(".", 1)[0] + "_." + filename.rsplit(".", 1)[1]
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed file: {old_file_path} to {new_file_path}")
    print("Renaming completed.")
except Exception as e:
    print(f"An error occurred: {e}")
    