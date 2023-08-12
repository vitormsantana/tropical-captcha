from PIL import Image
import cv2
import os
import numpy as np

input_folder = r"C:\Users\visantana\Documents\tropical-captcha\Captchas"
output_folder = r"C:\Users\visantana\Documents\tropical-captcha\Letters"

opencv_output_folder = os.path.join(output_folder, "OpenCV_Processed")

def apply_threshold(image, threshold_value):
    thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1]
    return thresholded

def apply_dilate(image):
    dilated = cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=1)
    return dilated

def apply_erode(image):
    eroded = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=1)
    return eroded

def detect_letter_positions(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letter_positions = []

    for contour in contours:
        x, _, w, _ = cv2.boundingRect(contour)
        letter_positions.append(x)

    letter_positions.sort()
    return letter_positions

def draw_contours(image, contours):
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)
    return image_with_contours

def generate_processed_images(image_path, opencv_output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Define threshold levels
    threshold_levels = [100, 150, 200, 250]  # Adjust as needed
    
    processed_images = []
    images_with_contours = []

    for threshold_value in threshold_levels:
        thresholded = apply_threshold(image, threshold_value)
        
        dilated_then_eroded = apply_erode(apply_dilate(thresholded))
        eroded_then_dilated = apply_dilate(apply_erode(thresholded))
        
        for i in range(8):
            if i < 4:
                treatment_name = "Threshold_Dilate_Erode"
                processed_image = apply_erode(apply_dilate(thresholded))
            else:
                treatment_name = "Threshold_Erode_Dilate"
                processed_image = apply_dilate(apply_erode(thresholded))

            contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_with_contours = draw_contours(processed_image, contours)
            
            processed_images.append((f"{image_name}_{i + 1}_{treatment_name}", processed_image))
            images_with_contours.append((f"{image_name}_{i + 1}_{treatment_name}_With_Contours", image_with_contours))

    # Save the processed images
    for i, (treatment_name, processed_image) in enumerate(processed_images):
        image_name_suffix = f"{image_name}_{i + 1}"
        image_path = os.path.join(opencv_output_folder, f"{image_name_suffix}_{treatment_name}.PNG")
        cv2.imwrite(image_path, processed_image)
        print(f"Saved processed image ({treatment_name}): {image_path}")

    # Save images with contours
    for i, (treatment_name, image_with_contours) in enumerate(images_with_contours):
        image_name_suffix = f"{image_name}_{i + 1}_{treatment_name}_With_Contours"
        image_path = os.path.join(opencv_output_folder, f"{image_name_suffix}.PNG")
        cv2.imwrite(image_path, image_with_contours)
        print(f"Saved image with contours ({treatment_name}): {image_path}")

def main():
    try:
        captcha_files = [f for f in os.listdir(input_folder) if f.endswith(".PNG")]
        for captcha_file in captcha_files:
            image_path = os.path.join(input_folder, captcha_file)
            generate_processed_images(image_path, opencv_output_folder)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
