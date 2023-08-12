import cv2
import os
import numpy as np

input_folder = r"C:\Users\visantana\Documents\tropical-captcha\Captchas"
output_folder = r"C:\Users\visantana\Documents\tropical-captcha"

opencv_output_folder = os.path.join(output_folder, "OpenCV_Processed")

def apply_threshold(image, threshold_value):
    thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1]
    return thresholded

def apply_dilate(image):
    dilated = cv2.dilate(image, np.ones((2, 3), np.uint8), iterations=1)
    return dilated

def apply_erode(image):
    eroded = cv2.erode(image, np.ones((3, 2), np.uint8), iterations=1)
    return eroded

def draw_contours(image, contours):
    image_with_contours = np.zeros_like(image)
    cv2.drawContours(image_with_contours, contours, -1, (255, 255, 255), 1)
    return image_with_contours

def crop_and_save_letters(image_path, letter_positions):
    image = cv2.imread(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    for i, (start_pos, end_pos) in enumerate(letter_positions):
        letter_image = image[:, start_pos:end_pos]
        letter_image_path = os.path.join(opencv_output_folder, f"{image_name}_letter_{i+1}.png")
        cv2.imwrite(letter_image_path, letter_image)
        print(f"Saved cropped letter image {i+1} for {image_name}: {letter_image_path}")

def generate_processed_images(image_path, opencv_output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Define threshold levels
    threshold_levels = [1, 15]  # Adjust as needed

    processed_images = []
    contours_images = []

    for threshold_value in threshold_levels:
        thresholded = apply_threshold(image, threshold_value)

        dilated_then_eroded = apply_erode(apply_dilate(thresholded))
        eroded_then_dilated = apply_dilate(apply_erode(thresholded))

        for i in range(1):
            if i < 4:
                treatment_name = "Threshold_Dilate_Erode"
                processed_image = apply_erode(apply_dilate(thresholded))
            else:
                treatment_name = "Threshold_Erode_Dilate"
                processed_image = apply_dilate(apply_erode(thresholded))

            # Construct the processed image filename
            image_name_suffix = f"{image_name}_{i + 1}.PNG"
            image_path = os.path.join(opencv_output_folder, image_name_suffix)
            
            # Save the processed image
            cv2.imwrite(image_path, processed_image)
            print(f"Saved processed image ({image_name_suffix}): {image_path}")


    # Save contour images
    #for i, (treatment_name, contours_image) in enumerate(contours_images):
    #    image_name_suffix = f"{image_name}_{i + 1}_{treatment_name}_Contours"
    #    image_path = os.path.join(opencv_output_folder, f"{image_name_suffix}.PNG")
    #    cv2.imwrite(image_path, contours_image)
    #    print(f"Saved contour image ({treatment_name}): {image_path}")

def main():
    try:
        captcha_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]
        for captcha_file in captcha_files:
            image_path = os.path.join(input_folder, captcha_file)
            generate_processed_images(image_path, opencv_output_folder)
            #crop_and_save_letters(image_path, [(0, 20), (21, 40)])  # Adjust letter positions as needed
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
