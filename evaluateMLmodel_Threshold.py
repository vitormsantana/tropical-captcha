from keras.models import load_model
import numpy as np
import os
import pickle
import cv2

# Load the model
model_path = r"C:\Users\visantana\Documents\tropical-captcha\captcha_model.h5"
model = load_model(model_path)

# Load the label binarizer
MODEL_LABELS_FILENAME = r"C:\Users\visantana\Documents\tropical-captcha\model_labels.pkl"
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Division positions
division_positions = [(5, 42), (37, 71), (67, 100), (100, 129), (130, 159), (160, 179)]

# Directory to save images
output_directory = r"C:\Users\visantana\Documents\tropical-captcha\testSet"

# Iterate through the image files
input_folder = r"C:\Users\visantana\Documents\tropical-captcha\testSet"
image_files = [f for f in os.listdir(input_folder) if f.startswith("captcha_") and f.endswith("_teste.png")]

for image_file in image_files:
    # Load and preprocess the original image
    original_image_path = os.path.join(input_folder, image_file)

    # Apply OpenCV processing to the original image
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(original_image, 15, 255, cv2.THRESH_BINARY)[1]
    processed_original_image = cv2.erode(cv2.dilate(thresholded_image, np.ones((2, 3), np.uint8), iterations=1), np.ones((3, 2), np.uint8), iterations=1)
    cv2.imwrite(os.path.join(output_directory, f"processed_original_{image_file}"), processed_original_image)

    # Resize the processed original image
    processed_original_image_resized = cv2.resize(processed_original_image, (180, 50))
    processed_original_image_array = processed_original_image_resized / 255.0

    # Add the channel axis for the processed original image
    processed_original_image_array = np.expand_dims(processed_original_image_array, axis=-1)

    # Process and save each cropped part
    for i, (start, end) in enumerate(division_positions):
        divided_image = processed_original_image_resized[:, start:end]

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

        print(f"Image {image_file} - Cropped Image {i+1} - Predicted Label: {predicted_label}")

        # Save the analyzed cropped image
        cropped_image_output_path = os.path.join(output_directory, f"crop_{image_file[:-10]}_{i+1}_predicted_{predicted_label}.png")
        cv2.imwrite(cropped_image_output_path, divided_image_resized)
