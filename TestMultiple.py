from keras.models import load_model
import numpy as np
import os
import pickle
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

def cut_bottom(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[:-cut_height, :]

def cut_top(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[cut_height:, :]

# Directory containing trained model files
models_directory = r'C:\Users\visantana\Documents\tropical-captcha\models'

# Load the label binarizer
MODEL_LABELS_FILENAME = r"C:\Users\visantana\Documents\tropical-captcha\model_labels.pkl"
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Division positions
division_positions = [(5, 42), (37, 71), (65, 105), (94, 133), (125, 159), (155, 179)]

# Directory to save images
output_directory = r"C:\Users\visantana\Documents\tropical-captcha\testSet"

# Load each model from the models directory
num_models = 1
trained_models = []  # List to store loaded models
for i in range(num_models):
    
    model_folders = [folder for folder in os.listdir(models_directory) if folder.startswith(f'paralelized_captcha_model_{i}')]
    
    if model_folders:
        model_folder = model_folders[0]  # Assuming there's only one matching folder
        print('model: ', model_folder)
        model_path = os.path.join(models_directory, model_folder, model_folder)
        
        # Load the model without specifying custom optimizer class
        loaded_model = load_model(model_path, compile=False)
        trained_models.append(loaded_model)
    else:
        print(f"Model folder for index {i} not found.")

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
        
            # Make predictions using the ensemble of models
            ensemble_predictions = np.mean([model.predict(divided_image_reshaped) for model in trained_models], axis=0)
            predicted_label_index = np.argmax(ensemble_predictions, axis=1)
            predicted_label = lb.classes_[predicted_label_index][0]
        
            captcha_word += predicted_label  # Collect the predicted letters
        
            # Save the analyzed cropped image
            cropped_image_output_path = os.path.join(output_directory, f"crop_{image_file[:-10]}_{i+1}_predicted_{predicted_label}.png")
            cv2.imwrite(cropped_image_output_path, divided_image_resized)
            
            # Create a heatmap of class probabilities
            plt.figure(figsize=(8, 6))
            heatmap = sns.heatmap(ensemble_predictions, annot=True, cmap='YlGnBu', xticklabels=lb.classes_, yticklabels=[predicted_label], fmt='.2f', cbar=False)
            plt.title(f"Heatmap for Image: {image_file}, Part {i+1}")
            plt.xlabel("Class")
            plt.ylabel("Predicted Label")
            
            # Save the heatmap as an image
            heatmap_output_path = os.path.join(output_directory, f"heatmap_{image_file[:-10]}_{i+1}.png")
            plt.savefig(heatmap_output_path)
            plt.close()
        
            # Find the indices of top and bottom predicted probabilities
            top_indices = np.argsort(ensemble_predictions[0])[::-1]
            bottom_indices = np.argsort(ensemble_predictions[0])
        
            # Get corresponding class labels
            top_labels = lb.classes_[top_indices]
            bottom_labels = lb.classes_[bottom_indices]
        
            # Print top and bottom predicted probabilities and labels
            print(f"Top predicted letters and probabilities: {top_labels[:5]} - {ensemble_predictions[0][top_indices][:5]}")
            print(f"Bottom predicted letters and probabilities: {bottom_labels[:5]} - {ensemble_predictions[0][bottom_indices][:5]}")
            
            # Plot the accuracy distribution per class
            plt.figure(figsize=(10, 6))
            sns.barplot(x=lb.classes_, y=ensemble_predictions[0])
            plt.title(f"Accuracy Distribution for Image: {image_file}, Part {i+1}")
            plt.xlabel("Class")
            plt.ylabel("Accuracy")
            plt.xticks(rotation=45)
            
            # Save the accuracy distribution plot as an image
            accuracy_dist_output_path = os.path.join(output_directory, f"accuracy_dist_{image_file[:-10]}_{i+1}.png")
            plt.savefig(accuracy_dist_output_path)
            plt.close()
            
        
        print(f"Image {image_file} - Predicted Captcha Word: {captcha_word}")
        
        # Save the pre-crop image with the captcha word prediction and green lines
        pre_crop_image_with_prediction_path = os.path.join(output_directory, f"pre_crop_{image_file[:-10]}_predicted_{captcha_word}.png")
        
        # Draw green lines at division positions
        for start, _ in division_positions:
            cv2.line(processed_original_image_resized, (start, 0), (start, 50), (0, 255, 0), 1)
        
        cv2.imwrite(pre_crop_image_with_prediction_path, processed_original_image_resized)
    print('--------------------------------------------------------------------------------------------------------------------------------------')
        
print("test completed.")
        
        
