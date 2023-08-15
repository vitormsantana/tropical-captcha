from keras.models import load_model
import numpy as np
import os
import pickle
import cv2
import pandas as pd
import pandas as pd
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

# Load the model
model_path = r"C:\Users\visantana\Documents\tropical-captcha\captcha_model_SimpleCNN.h5"
model = load_model(model_path, compile=False)

# Load the label binarizer
MODEL_LABELS_FILENAME = r"C:\Users\visantana\Documents\tropical-captcha\model_labels.pkl"
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Division positions
division_positions = [(5, 42), (37, 71), (65, 105), (94, 133), (125, 159), (155, 179)]

# Directory containing labeled captchas
labeled_folder = r"C:\Users\visantana\Documents\tropical-captcha\labeled_testSet"


# Initialize the results table and statistics
results = []
statistics = []

# Iterate through the labeled captchas
for labeled_image_file in os.listdir(labeled_folder):
    if labeled_image_file.endswith(".png"):
        # Load and preprocess the labeled image
        labeled_image_path = os.path.join(labeled_folder, labeled_image_file)
        labeled_word = labeled_image_file[:-4]  # Remove file extension
        
        # Apply OpenCV processing to the labeled image
        labeled_image = cv2.imread(labeled_image_path, cv2.IMREAD_GRAYSCALE)
        thresholded_image = cv2.threshold(labeled_image, 15, 255, cv2.THRESH_BINARY)[1]
        processed_labeled_image = cv2.erode(cv2.dilate(thresholded_image, np.ones((2, 3), np.uint8), iterations=1), np.ones((3, 2), np.uint8), iterations=1)
        
        # Resize the processed labeled image
        processed_labeled_image_resized = cv2.resize(processed_labeled_image, (180, 50))
        processed_labeled_image_array = processed_labeled_image / 255.0
        
        # Add the channel axis for the processed labeled image
        processed_labeled_image_array = np.expand_dims(processed_labeled_image_array, axis=-1)
        
        # Process the labeled image and predict the word
        predicted_word = ""
        for i, (start, end) in enumerate(division_positions):
            divided_image = processed_labeled_image_resized[:, start:end]
            divided_image = cut_top(divided_image, 0.18)
            divided_image = cut_bottom(divided_image, 0.23)
            new_width = 26
            new_height = 31
            divided_image_resized = cv2.resize(divided_image, (new_width, new_height))
            divided_image_reshaped = divided_image_resized.reshape((1, new_height, new_width, 1))
            predicted_probs = model.predict(divided_image_reshaped)
            predicted_label_index = np.argmax(predicted_probs, axis=1)
            predicted_label = lb.classes_[predicted_label_index][0]
            predicted_word += predicted_label
        
        # Compare predicted word with labeled word
        n_correct = sum(c1 == c2 for c1, c2 in zip(predicted_word, labeled_word))
        n_incorrect = len(labeled_word) - n_correct
        percent_correct = (n_correct / len(labeled_word)) * 100
        
        # Create a dictionary to store the results
        result = {"Labeled Word": labeled_word, "Predicted Word": predicted_word, 
                  "n_correct": n_correct, "n_incorrect": n_incorrect, 
                  "%correct": percent_correct}
        
        # Append the character columns
        for char in lb.classes_:
            if char in labeled_word:
                if char in predicted_word:
                    result[char] = "c"
                else:
                    result[char] = "u"
            else:
                result[char] = ""  # Empty cell for characters not in the image
            
        results.append(result)
        
        # Create a dictionary for the statistics
        statistic = {"Labeled Word": labeled_word, "Correct Ratio": percent_correct}
        statistics.append(statistic)

# Create a DataFrame from the results and save it as a CSV file
results_df = pd.DataFrame(results)
results_csv_path = r"C:\Users\visantana\Documents\tropical-captcha\result_statistics.csv"
results_df.to_csv(results_csv_path, index=False)


# Extract the columns for characters (a-z, 1-9)
char_columns = [col for col in results_df.columns if col.isalnum()]

# Convert "c" and "u" values to 1 and 0
results_df[char_columns] = results_df[char_columns].applymap(lambda x: 1 if x == "c" else 0)

# Calculate the ratio of correct predictions for each character
results_df["Correct Ratio"] = results_df[char_columns].sum(axis=1) / len(char_columns)

# Plot the bar graph
plt.figure(figsize=(12, 6))
results_df.plot(y="Correct Ratio", kind="bar", color="skyblue")
plt.title("Correct Prediction Ratio for Each Image")
plt.xlabel("Image")
plt.ylabel("Correct Ratio")
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
plt.tight_layout()
plt.show()
