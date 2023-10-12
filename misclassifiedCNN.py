from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3, MobileNetV2  # Import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import os
import cv2

misclassified_images = []
misclassified_labels = []

# Define the paths to your training data folders
data_folder = r"C:\Users\visantana\Documents\tropical-captcha\Letters"
MODEL_LABELS_FILENAME = "model_labels.pkl"

# Define the target size for the resized images
target_size = (32, 32)


# Create an ImageDataGenerator to load and preprocess images from folders
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.30)

# Load and preprocess training data from the folders
train_data = datagen.flow_from_directory(
    data_folder,
    target_size=target_size,
    color_mode="grayscale",  # Specify grayscale here
    class_mode="categorical",
    subset="training",
    batch_size=32,
    shuffle=True,
    seed=42
)

# Load and preprocess validation data from the folders
val_data = datagen.flow_from_directory(
    data_folder,
    target_size=target_size,
    color_mode="grayscale",  # Specify grayscale here
    class_mode="categorical",
    subset="validation",
    batch_size=32,
    shuffle=False
)

# Convert the class indices to class labels
class_labels = list(train_data.class_indices.keys())

# Convert the labels (letters) into one-hot encodings that Keras can work with
lb = LabelBinarizer().fit(class_labels)

# Save the mapping from labels to one-hot encodings
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Define the list of model names you want to use
model_names = ['SimpleCNN']  # You can add more model names here

for model_name in model_names:
    # Create the neural network model
    if model_name == 'SimpleCNN':
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation="relu", input_shape=target_size + (1,)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.5))
        model.add(Dense(35, activation="softmax"))
    else:
        continue
    
    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Train the neural network with class weights
    history = model.fit(train_data, validation_data=val_data, epochs=17, verbose=1)
    
    # Save the model weights
    model.save("captcha_model_SimpleCNN.h5")

    # Visualize accuracy and loss curves
    plt.figure(figsize=(10, 6))
    # ... (plot accuracy and loss curves, as in your previous code)

    # Get the complete test dataset
    X_test = []
    Y_test = []
    for batch in val_data:
        X_test.extend(batch[0])
        Y_test.extend(batch[1])
        if len(Y_test) >= len(val_data.classes):
            break
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    # Predict the labels using the trained model
    Y_pred = model.predict(X_test)
    Y_pred_labels = np.argmax(Y_pred, axis=1)  # Convert one-hot to class labels
    Y_true_labels = np.argmax(Y_test, axis=1)   # Convert one-hot to class labels
    
    # Identify and store misclassified images
    for i in range(len(Y_true_labels)):
        if Y_true_labels[i] != Y_pred_labels[i]:
            misclassified_image = X_test[i] * 255  # Scaling back to [0, 255] range
            misclassified_images.append(misclassified_image)
            misclassified_labels.append(lb.classes_[Y_true_labels[i]][0])
            
    misclassified_folder = r"C:\Users\visantana\Documents\tropical-captcha\Misclassified_Images"

    # Create the folder if it doesn't exist
    os.makedirs(misclassified_folder, exist_ok=True)
    
    # Save misclassified images
    for i, (image, true_label) in enumerate(zip(misclassified_images, misclassified_labels)):
        position = i % 6 + 1  # Calculate position as 1 to 6
        image_filename = f"misclassified_{i}_pos_{position}_true_{true_label}.png"
        image_path = os.path.join(misclassified_folder, image_filename)
        cv2.imwrite(image_path, image)

    # Compute confusion matrix
    conf_matrix = confusion_matrix(Y_true_labels, Y_pred_labels)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Count wrong predictions per position
    wrong_predictions_per_position = [0] * 6  # Initialize the counts to 0 for each position
    for i, (true_label, pred_label) in enumerate(zip(Y_true_labels, Y_pred_labels)):
        if true_label != pred_label:
            position = i % 6  # Calculate position as 0 to 5
            wrong_predictions_per_position[position] += 1

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    positions = np.arange(1, 7)  # Positions 1 to 6
    plt.bar(positions, wrong_predictions_per_position, align='center')
    plt.xlabel('Position')
    plt.ylabel('Number of Wrong Predictions')
    plt.title('Wrong Predictions per Position')
    plt.xticks(positions)
    plt.show()

    # Evaluate the model on the test data
    evaluation = model.evaluate(X_test, Y_test)
    print("Model:", model_name)
    print("Test Loss:", evaluation[0])
    print("Test Accuracy:", evaluation[1])
    
    # Check for overfitting
    if history.history['val_loss'][-1] > history.history['val_loss'][0]:
        print("The model might be overfitting.")
    else:
        print("The model seems to be performing well.")