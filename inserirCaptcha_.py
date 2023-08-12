# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 21:03:31 2023

@author: visantana
"""
#Renomear inserindo 'captcha_'

folder_path_2 = r"C:\Users\visantana\Documents\tropical-captcha\Captchas"

import os

try:
    for filename in os.listdir(folder_path_2):
        if not filename.startswith("captcha_"):
            new_filename = f"captcha_{filename}"
            old_file_path = os.path.join(folder_path_2, filename)
            new_file_path = os.path.join(folder_path_2, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed file: {old_file_path} to {new_file_path}")
    print("Renaming completed.")
except Exception as e:
    print(f"An error occurred: {e}")