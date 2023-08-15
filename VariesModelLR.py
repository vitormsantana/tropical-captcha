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

# Define the paths to your training data folders
data_folder = r"C:\Users\visantana\Documents\tropical-captcha\Letters"
MODEL_LABELS_FILENAME = "model_labels.pkl"

# Create an ImageDataGenerator to load and preprocess images from folders
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.25)

# Load and preprocess training data from the folders
train_data = datagen.flow_from_directory(
    data_folder,
    target_size=(31, 26),
    color_mode="grayscale",
    class_mode="categorical",
    subset="training",
    batch_size=32,
    shuffle=True,
    seed=42
)

# Load and preprocess validation data from the folders
val_data = datagen.flow_from_directory(
    data_folder,
    target_size=(31, 26),
    color_mode="grayscale",
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

# Create the neural network model
model = Sequential()

# First convolutional layer with max pooling, L2 regularization, and dropout
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(31, 26, 1), activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.5))  # Adjust dropout rate as needed

# Hidden layer with 500 nodes, L2 regularization, and dropout
model.add(Flatten())
model.add(Dense(500, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))  # Adjust dropout rate as needed

# Output layer with 35 nodes (one for each possible letter/number we predict)
model.add(Dense(35, activation="softmax"))

# Ask Keras to build the TensorFlow model behind the scenes
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Calculate the total number of training images
total_training_images = len(train_data.filenames)

# Set the desired interval for displaying accuracy
display_interval = 200  # Display accuracy every 200 images

# Train the neural network
history = model.fit(train_data, validation_data=val_data, epochs=17, verbose=1, steps_per_epoch=total_training_images // train_data.batch_size)

# Save the model weights
model.save("captcha_model.h5")

# Get the complete training dataset
X_train = []
Y_train = []
for batch in train_data:
    X_train.extend(batch[0])
    Y_train.extend(batch[1])
    if len(X_train) >= len(train_data.classes):
        break
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Get the complete validation dataset
X_test = []
Y_test = []
for batch in val_data:
    X_test.extend(batch[0])
    Y_test.extend(batch[1])
    if len(Y_test) >= len(val_data.classes):
        break
X_test = np.array(X_test)
Y_test = np.array(Y_test)

# Evaluate the model on the test data
evaluation = model.evaluate(X_test, Y_test)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

# Plotting the epoch vs. accuracy and loss
plt.figure(figsize=(10, 6))

# Plot accuracy
plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Epoch vs. Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Plot loss
plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Epoch vs. Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Sort class labels in alphabetical order
class_labels.sort()

# Get the predictions and true labels for the validation data
val_data.reset()  # Reset the generator to the beginning
Y_pred = model.predict(val_data)
Y_pred_labels = np.argmax(Y_pred, axis=1)
Y_true_labels = val_data.classes

# Compute confusion matrix
conf_matrix = confusion_matrix(Y_true_labels, Y_pred_labels)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Loop through training batches
for i, batch in enumerate(train_data, 1):
    X_batch, Y_batch = batch[0], batch[1]

    # Train the model with the current batch
    history_batch = model.train_on_batch(X_batch, Y_batch)

    # Display accuracy every display_interval images
    if i % (display_interval // train_data.batch_size) == 0:
        print(f"Trained {i * train_data.batch_size}/{total_training_images} images | Batch Loss: {history_batch[0]:.4f} | Batch Accuracy: {history_batch[1]:.4f}")

    # Stop the loop if all images have been trained
    if i * train_data.batch_size >= total_training_images:
        break