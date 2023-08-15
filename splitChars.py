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

def generate_processed_images(image, opencv_output_folder):
    image_name = os.path.splitext(os.path.basename(image))[0]
    gray_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Define threshold levels
    threshold_levels = [1, 15]  # Adjust as needed

    for threshold_value in threshold_levels:
        thresholded = apply_threshold(gray_image, threshold_value)

        dilated_then_eroded = apply_erode(apply_dilate(thresholded))
        eroded_then_dilated = apply_dilate(apply_erode(thresholded))

        for i in range(2):
            if i < 4:
                processed_image = apply_erode(apply_dilate(thresholded))
            else:
                processed_image = apply_dilate(apply_erode(thresholded))

            processed_image_path = os.path.join(opencv_output_folder, f"{image_name}_{i + 1}.png")
            cv2.imwrite(processed_image_path, processed_image)
            print(f"Saved processed image ({image_name}_{i + 1}.png): {processed_image_path}")

def main():
    try:
        os.makedirs(output_folder_divided, exist_ok=True)
        os.makedirs(output_folder_letters, exist_ok=True)
        os.makedirs(opencv_output_folder, exist_ok=True)
        
        labeled_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

        division_positions = [(5, 42), (37, 71), (67, 100), (100, 129), (130, 159), (160, 188)]
        
        for labeled_file in labeled_files:
            image_path = os.path.join(input_folder, labeled_file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            labeled_word = image_name
            
            divided_images = divide_image(image, division_positions)
            
            for letter_position, divided_image in enumerate(divided_images, start=1):
                divided_image = cut_top(divided_image, 0.18)
                divided_image = cut_bottom(divided_image, 0.23)  # Cut 28% from the bottom
                character = labeled_word[letter_position - 1]
                
                # Save divided image in the respective character folder
                char_folder = os.path.join(output_folder_letters, character)
                os.makedirs(char_folder, exist_ok=True)
                char_image_path = os.path.join(char_folder, f"{image_name}_{letter_position}_{character}.png")
                cv2.imwrite(char_image_path, divided_image)
                print(f"Saved divided letter image {letter_position} for word {labeled_word}: {char_image_path}")
                
                # Save the same image in the Divided Captchas folder
                divided_image_path = os.path.join(output_folder_divided, f"4lines_{image_name}_{letter_position}_{character}.png")
                cv2.imwrite(divided_image_path, divided_image)
                print(f"Saved divided letter image {letter_position} for word {labeled_word} in divided_captchas: {divided_image_path}")
                
            
            generate_processed_images(image_path, opencv_output_folder)  # Apply processing to the image
            # Copy unclassified image to the Divided Captchas folder
            unclassified_image_path = os.path.join(input_folder, labeled_file)
            unclassified_image_dest = os.path.join(output_folder_divided, labeled_file)
            
            # Check if the file already exists in the destination folder
            if not os.path.exists(unclassified_image_dest):
                shutil.copy2(unclassified_image_path, unclassified_image_dest)
                print(f"Copied unclassified image {labeled_file} to {output_folder_divided}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
