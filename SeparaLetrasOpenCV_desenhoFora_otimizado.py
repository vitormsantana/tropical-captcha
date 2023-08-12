import cv2
import os

input_folder = r"C:\Users\visantana\Documents\tropical-captcha\OpenCV_Processed"
output_folder = r"C:\Users\visantana\Documents\tropical-captcha\Divided_Captchas"

def divide_image(image, division_positions):
    divided_images = []
    for start_pos, end_pos in division_positions:
        divided_images.append(image[:, start_pos:end_pos])
    return divided_images

def draw_division_lines(image, positions):
    image_with_divisions = image.copy()
    for position in positions:
        cv2.line(image_with_divisions, (position, 0), (position, image.shape[0]), (0, 255, 0), 2)
    return image_with_divisions

def cut_bottom(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[:-cut_height, :]

def cut_top(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[cut_height:, :]

def main():
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        processed_files = [f for f in os.listdir(input_folder) if f.endswith(".PNG")]

        for processed_file in processed_files:
            image_path = os.path.join(input_folder, processed_file)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            image_width = image.shape[1]
            captcha_number = image_name.split("_")[1]
            
            division_positions = [(5, 42), (37, 71), (67, 100), (100, 129), (130, 159), (160, 188)]
            
            divided_images = divide_image(image, division_positions)
            for version, divided_image in enumerate(divided_images, start=1):
                divided_image = cut_top(divided_image, 0.18)
                divided_image = cut_bottom(divided_image, 0.23)  # Cut 28% from the bottom
                for letter_position in range(1, 7):
                    divided_image_path = os.path.join(output_folder, f"captcha_{captcha_number}_{version}_{letter_position}.PNG")
                    #cv2.imwrite(divided_image_path, divided_image)
                    print(f"Saved divided image {letter_position} for captcha {captcha_number}, version {version}: {divided_image_path}")
                    
                    # Draw division lines on the divided image
                    image_with_divisions = draw_division_lines(divided_image, [pos[1] for pos in division_positions[:-1]])
                    divided_image_with_lines_path = os.path.join(output_folder, f"captcha_{captcha_number}_{version}_{letter_position}_lines.PNG")
                    cv2.imwrite(divided_image_with_lines_path, image_with_divisions)
                    print(f"Saved divided image {letter_position} with lines for captcha {captcha_number}, version {version}: {divided_image_with_lines_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
