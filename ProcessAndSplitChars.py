import cv2
import os
import shutil
import numpy as np

input_folder = r"C:\Users\visantana\Documents\tropical-captcha\labeled_testSet"
output_folder_divided = r"C:\Users\visantana\Documents\tropical-captcha\Divided_Captchas"
output_folder_letters = r"C:\Users\visantana\Documents\tropical-captcha\Letters"
opencv_output_folder = r"C:\Users\visantana\Documents\tropical-captcha\OpenCV_Processed"

def divide_image(image, division_positions):
    divided_images = []
    for start_pos, end_pos in division_positions:
        divided_images.append(image[:, start_pos:end_pos])
    return divided_images

def cut_bottom(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[:-cut_height, :]

def cut_top(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[cut_height:, :]

def apply_threshold(image, threshold_value):
    thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1]
    return thresholded

def apply_dilate(image):
    dilated = cv2.dilate(image, np.ones((2, 3), np.uint8), iterations=1)
    return dilated

def apply_erode(image):
    eroded = cv2.erode(image, np.ones((3, 2), np.uint8), iterations=1)
    return eroded

def delete_files_with_suffix(folder_path, suffixes):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            for suffix in suffixes:
                if file.lower().endswith(suffix.lower()):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

def generate_processed_images(image):
    # Process the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold
    threshold_value = 15
    thresholded = apply_threshold(gray_image, threshold_value)
    # Apply dilate and erode
    processed_image = apply_erode(apply_dilate(thresholded))
    return processed_image

def main():
    try:
        os.makedirs(output_folder_divided, exist_ok=True)
        os.makedirs(output_folder_letters, exist_ok=True)
        os.makedirs(opencv_output_folder, exist_ok=True)
        
        labeled_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]
        
        division_positions = [(5, 42), (37, 71), (67, 100), (100, 129), (130, 159), (160, 188)]
        
        new_files_written_divided = 0
        existing_files_divided = 0
        new_files_written_letters = 0
        existing_files_letters = 0
        
        for labeled_file in labeled_files:
            image_path = os.path.join(input_folder, labeled_file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Apply processing to the image
            processed_image = generate_processed_images(image)
            
            # Save the processed image before cropping
            processed_image_path = os.path.join(opencv_output_folder, f"processed_{image_name}.png")
            cv2.imwrite(processed_image_path, processed_image)
            
            # Divide the processed image
            divided_images = divide_image(processed_image, division_positions)
            
            for letter_position, divided_image in enumerate(divided_images, start=1):
                divided_image = cut_top(divided_image, 0.18)
                divided_image = cut_bottom(divided_image, 0.23)  # Cut 28% from the bottom
                
                # Save divided image in the respective character folder
                char_folder = os.path.join(output_folder_letters, image_name[letter_position - 1])
                os.makedirs(char_folder, exist_ok=True)
                char_image_path = os.path.join(char_folder, f"{image_name}_{letter_position}_{image_name[letter_position - 1]}.png")
                
                # Check if the file already exists in letters folder
                if os.path.exists(char_image_path):
                    existing_files_letters += 1
                else:
                    cv2.imwrite(char_image_path, divided_image)
                    new_files_written_letters += 1
                
                # Save the same image in the Divided Captchas folder, replacing if it already exists
                divided_image_path = os.path.join(output_folder_divided, f"4lines_{image_name}_{letter_position}_{image_name[letter_position - 1]}.png")
                
                # Check if the file already exists in divided_captchas folder
                if os.path.exists(divided_image_path):
                    existing_files_divided += 1
                else:
                    cv2.imwrite(divided_image_path, divided_image)
                    new_files_written_divided += 1
                        
        print(f"Number of new files written in divided_captchas: {new_files_written_divided}")
        print(f"Number of existing files in divided_captchas: {existing_files_divided}")
        print(f"Number of new files written in /letters: {new_files_written_letters}")
        print(f"Number of existing files in /letters: {existing_files_letters}")
                
    except Exception as e:               
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
