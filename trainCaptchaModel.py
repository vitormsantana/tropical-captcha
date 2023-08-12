from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from keras import regularizers
from keras.layers import Dropout

# Define the paths to your training data folders
data_folder = r"C:\Users\visantana\Documents\tropical-captcha\Letters"
MODEL_LABELS_FILENAME = "model_labels.pkl"

# Create an ImageDataGenerator to load and preprocess images from folders
datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.30)

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

# Train the neural network
model.fit(train_data, validation_data=val_data, epochs=37, verbose=1)

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