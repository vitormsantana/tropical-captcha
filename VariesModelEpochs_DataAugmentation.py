from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import pickle
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.metrics import roc_curve, auc
from keras.layers import Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import Model
from sklearn.inspection import permutation_importance
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow as tf
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

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

# Use the same LabelBinarizer instance for transforming labels
lb = LabelBinarizer().fit(class_labels)

# ... Rest of your code ...

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
#evaluation = model.evaluate(X_test, Y_test)
#@print("Test Loss:", evaluation[0])
#rint("Test Accuracy:", evaluation[1])

# Predict probabilities for validation data
#Y_pred_probs = model.predict(X_test)

# Convert predicted probabilities to class labels
#Y_pred = np.argmax(Y_pred_probs, axis=1)

# Convert one-hot encoded labels back to original labels
#Y_true_labels = lb.inverse_transform(Y_test)
#Y_pred_labels = lb.classes_[Y_pred]

# Compute precision and recall values for each class
precision = dict()
recall = dict()
#for i in range(len(class_labels)):
#    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], Y_pred_probs[:, i])

# Define the neural network model as a function
# Create the neural network model
def create_model(filters=20, units=500, dropout_rate=0.5):
    model = Sequential()
    model.add(Conv2D(filters=filters, kernel_size=(5, 5), padding="same", input_shape=(31, 26, 1), activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(units=units, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(34, activation="softmax"))  # Here, change 35 to the actual number of classes
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Create a KerasClassifier based on your model-building function
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define the parameter grid for the successive halving search
param_grid = {
    'filters': [20, 32, 64],
    'units': [250, 500, 750],
    'dropout_rate': [0.3, 0.5, 0.7]
}

# Define your custom scoring function
def custom_scoring(estimator, X, y):
    y_pred = estimator.predict(X)
    return accuracy_score(y, y_pred)

# Create your HalvingGridSearchCV
halving_grid_search = HalvingGridSearchCV(
    model, param_grid, cv=5, scoring=custom_scoring, factor=2, resource='n_samples'
)

# Fit the HalvingGridSearchCV
halving_grid_result = halving_grid_search.fit(X_train, np.argmax(Y_train, axis=1))

# Print the best parameters and the corresponding accuracy
print("Best parameters found: ", halving_grid_result.best_params_)
print("Best accuracy found: ", halving_grid_result.best_score_)