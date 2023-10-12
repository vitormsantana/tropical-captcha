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
                
        for labeled_file in labeled_files:
            image_path = os.path.join(input_folder, labeled_file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Apply processing to the image
            processed_image = generate_processed_images(image)
            
            # Save the processed image before cropping
            processed_image_path = os.path.join(opencv_output_folder, f"processed_{image_name}.png")
            cv2.imwrite(processed_image_path, processed_image)
                                        
    except Exception as e:               
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
