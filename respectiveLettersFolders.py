import os
import shutil
from PIL import Image

divided_captchas_folder = r"C:\Users\visantana\Documents\tropical-captcha\Divided_Captchas"
letters_folder = r"C:\Users\visantana\Documents\tropical-captcha\Letters"

try:
    letter_set = set()  # To store unique last letters
    
    # Collect unique last letters
    for filename in os.listdir(divided_captchas_folder):
        if filename.lower().endswith(".png"):
            letter = filename[-5].lower()  # Convert the last character to lowercase
            letter_set.add(letter)
    
    # Create folders for each letter and copy files
    for letter in letter_set:
        letter_folder_path = os.path.join(letters_folder, letter)
        
        if not os.path.exists(letter_folder_path):
            os.makedirs(letter_folder_path)
            print(f"Created folder: {letter_folder_path}")
        
        for filename in os.listdir(divided_captchas_folder):
            if filename.lower().endswith(".png") and filename[-5].lower() == letter:  # Convert the last character to lowercase
                source_file_path = os.path.join(divided_captchas_folder, filename)
                destination_file_path = os.path.join(letter_folder_path, filename)
                
                # Open the image, resize it to 31x26, and save it
                img = Image.open(source_file_path)
                resized_img = img.resize((26, 31), Image.ANTIALIAS)
                resized_img.save(destination_file_path)
                
                print(f"Copied and resized file: {source_file_path} to {destination_file_path}")

    print("Copying and resizing files completed.")
except Exception as e:
    print(f"An error occurred: {e}")
