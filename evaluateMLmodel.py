from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os
import pickle

# Load the model
model_path = r"C:\Users\visantana\Documents\tropical-captcha\captcha_model.h5"
model = load_model(model_path)

# Load the label binarizer
MODEL_LABELS_FILENAME = r"C:\Users\visantana\Documents\tropical-captcha\model_labels.pkl"
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Division positions
division_positions = [(5, 42), (37, 71), (67, 100), (100, 129), (130, 159), (160, 188)]

# Directory to save images
output_directory = r"C:\Users\visantana\Documents\tropical-captcha\testSet"

# Load and preprocess the original image
original_image_path = r"C:\Users\visantana\Documents\tropical-captcha\testSet\captcha_1_teste.png"
original_image = image.load_img(original_image_path, color_mode="grayscale", target_size=(50, 180))
original_image_array = image.img_to_array(original_image)
original_image_array /= 255.0

# Save the resized original image
resized_original_image_output_path = os.path.join(output_directory, "insomnia_resized.png")
resized_original_image_pil = image.array_to_img(original_image_array)
resized_original_image_pil.save(resized_original_image_output_path)

# Process and save each cropped part
for i, (start, end) in enumerate(division_positions):
    divided_image = original_image_array[:, start:end]

    # Resize the cropped image to the same size as the model
    new_width = 26
    new_height = 31
    divided_image_resized = image.array_to_img(divided_image)
    divided_image_resized = divided_image_resized.resize((new_width, new_height))
    divided_image_resized_array = image.img_to_array(divided_image_resized)

    # Add the channel axis
    divided_image_channel = np.expand_dims(divided_image_resized_array, axis=0)

    # Reshape the image for model prediction
    divided_image_channel = divided_image_channel.reshape((1, new_height, new_width, 1))

    # Make a prediction using the model
    predicted_probs = model.predict(divided_image_channel)
    predicted_label_index = np.argmax(predicted_probs, axis=1)
    predicted_label = lb.classes_[predicted_label_index][0]

    print(f"Cropped Image {i+1} - Predicted Label: {predicted_label}")

    # Save the analyzed cropped image
    cropped_image_pil = image.array_to_img(divided_image_resized_array)
    cropped_image_output_path = os.path.join(output_directory, f"crop_{i+1}_predicted_{predicted_label}.png")
    cropped_image_pil.save(cropped_image_output_path)

# Save the resized original image with the name "insomnia.png"
resized_original_image_pil.save(os.path.join(output_directory, "insomnia.png"))
