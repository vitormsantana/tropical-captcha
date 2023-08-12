from keras.models import load_model
import numpy as np
import os
import pickle
import cv2

def cut_bottom(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[:-cut_height, :]

def cut_top(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[cut_height:, :]

# Load the model
model_path = r"C:\Users\visantana\Documents\tropical-captcha\captcha_model.h5"
model = load_model(model_path)

# Load the label binarizer
MODEL_LABELS_FILENAME = r"C:\Users\visantana\Documents\tropical-captcha\model_labels.pkl"
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Division positions
division_positions = [(5, 42), (37, 71), (65, 105), (94, 133), (125, 159), (155, 179)]

#oi

# Directory to save images
output_directory = r"C:\Users\visantana\Documents\tropical-captcha\testSet"

# Iterate through the image files
input_folder = r"C:\Users\visantana\Documents\tropical-captcha\testSet"
image_files = [f for f in os.listdir(input_folder) if f.startswith("captcha_") and (f.endswith("_teste.png") or f.endswith("_teste.PNG"))]

for image_file in image_files:
    # Load and preprocess the original image
    original_image_path = os.path.join(input_folder, image_file)

    # Apply OpenCV processing to the original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(original_image, 15, 255, cv2.THRESH_BINARY)[1]
    processed_original_image = cv2.erode(cv2.dilate(thresholded_image, np.ones((2, 3), np.uint8), iterations=1), np.ones((3, 2), np.uint8), iterations=1)

    # Resize the processed original image
    processed_original_image_resized = cv2.resize(processed_original_image, (180, 50))
    processed_original_image_array = processed_original_image / 255.0
    
    # Add the channel axis for the processed original image
    processed_original_image_array = np.expand_dims(processed_original_image_array, axis=-1)
    
    # Process and save each cropped part
    captcha_word = ""
    for i, (start, end) in enumerate(division_positions):
        divided_image = processed_original_image_resized[:, start:end]
    
        # Apply cut_top and cut_bottom to improve the cropped image
        divided_image = cut_top(divided_image, 0.18)
        divided_image = cut_bottom(divided_image, 0.23)
    
        # Resize the cropped image to the same size as the model
        new_width = 26
        new_height = 31
        divided_image_resized = cv2.resize(divided_image, (new_width, new_height))

        
        # Reshape the image for model prediction
        divided_image_reshaped = divided_image_resized.reshape((1, new_height, new_width, 1))
    
        # Make a prediction using the model
        predicted_probs = model.predict(divided_image_reshaped)
        predicted_label_index = np.argmax(predicted_probs, axis=1)
        predicted_label = lb.classes_[predicted_label_index][0]
    
        captcha_word += predicted_label  # Collect the predicted letters
    
        # Save the analyzed cropped image
        cropped_image_output_path = os.path.join(output_directory, f"crop_{image_file[:-10]}_{i+1}_predicted_{predicted_label}.png")
        #Scv2.imwrite(cropped_image_output_path, divided_image_resized)
    
    print(f"Image {image_file} - Predicted Captcha Word: {captcha_word}")
    
    # Save the pre-crop image with the captcha word prediction and green lines
    pre_crop_image_with_prediction_path = os.path.join(output_directory, f"pre_crop_{image_file[:-10]}_predicted_{captcha_word}.png")
    
    # Draw green lines at division positions
    for start, _ in division_positions:
        cv2.line(processed_original_image_resized, (start, 0), (start, 50), (0, 255, 0), 1)
    
    cv2.imwrite(pre_crop_image_with_prediction_path, processed_original_image_resized)
    
    print("Process completed.")
