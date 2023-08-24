import base64
import json
import cv2
import numpy as np
import tensorflow as tf
import boto3
import pickle
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow GPU-related logs

# Initialize the S3 client
s3 = boto3.client('s3')

# Specify the bucket name and model file name
bucketName = 'tropical-captcha-bucket'
MODEL_FILENAME = 'captcha_model_SimpleCNN.h5'
s3_key = 'tropical-captcha/LambdaFiles/' + MODEL_FILENAME

# Download the model file from S3 to the local /tmp directory
local_model_path = '/tmp/' + MODEL_FILENAME
s3.download_file(bucketName, s3_key, local_model_path)

# Load the Keras model from the downloaded file
model = tf.keras.models.load_model(local_model_path)

# Specify the bucket name and label binarizer file name
bucketName = 'tropical-captcha-bucket'
MODEL_LABELS_FILENAME = 'model_labels.pkl'
s3_key_labels = 'tropical-captcha/LambdaFiles/' + MODEL_LABELS_FILENAME

# Download the label binarizer file from S3 to the local /tmp directory
local_labels_path = '/tmp/' + MODEL_LABELS_FILENAME

s3.download_file(bucketName, s3_key_labels, local_labels_path)

####################################################################################################################

# Load the label binarizer from the downloaded file
with open(local_labels_path, 'rb') as f:
    lb = pickle.load(f)

def cut_bottom(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[:-cut_height, :]

def cut_top(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[cut_height:, :]

# Division positions
division_positions = [(5, 42), (37, 71), (65, 105), (94, 133), (125, 159), (155, 179)]

def preprocess_image(image_data):

    # Extract the base64 encoded image data from the input string
    encoded_prefix = "data:image/png;base64,"
    if image_data.startswith(encoded_prefix):
        image_data = image_data[len(encoded_prefix):]
        
    decoded_image = base64.b64decode(image_data)
       
    print("Entering preprocess_image function")
    
    # Decode base64-encoded image
    nparr = np.frombuffer(decoded_image, np.uint8)
    print('nparr: ', nparr)
    
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    print('image: ', image)
    
    thresholded_image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)[1]
    print('thresholded_image: ', thresholded_image)
    
    processed_labeled_image = cv2.erode(cv2.dilate(thresholded_image, np.ones((2, 3), np.uint8), iterations=1), np.ones((3, 2), np.uint8), iterations=1)
        
    # Resize the processed labeled image
    processed_labeled_image_resized = cv2.resize(processed_labeled_image, (180, 50))
    processed_labeled_image_array = processed_labeled_image / 255.0
        
    # Add the channel axis for the processed labeled image
    processed_labeled_image_array = np.expand_dims(processed_labeled_image_array, axis=-1)
        
    # Process the labeled image and predict the word
    predicted_word = ""
    predicted_letters = []  # Initialize an empty list to store predicted letters
    
    for i, (start, end) in enumerate(division_positions):
        divided_image = processed_labeled_image_resized[:, start:end]
        divided_image = cut_top(divided_image, 0.18)
        divided_image = cut_bottom(divided_image, 0.23)
        new_width = 26
        new_height = 31
        
        divided_image_resized = cv2.resize(divided_image, (32, 32))  # Resize to match input size
        divided_image_reshaped = divided_image_resized.reshape((1, 32, 32, 1))  # Reshape for input
        
        prediction = model.predict(divided_image_reshaped)
        predicted_class = get_predicted_class(prediction)
        predicted_letters.append(predicted_class)  # Append the predicted letter to the list
        
    #print("Predicted letters1:", predicted_letters)
    return predicted_letters
 

def get_predicted_class(prediction):
    # Process the prediction and get the predicted class
    predicted_label_index = np.argmax(prediction, axis=1)
    predicted_label = lb.classes_[predicted_label_index][0]
    return predicted_label

def lambda_handler(event, context):
    print("Received event: %s", str(event)[:62])
    try:
        print("entrou no Try")
        logger.debug("Received event: %s", str(event)[:62])
        
        # Decode the base64-encoded image data from the input
        encoded_image = event['encoded_image']
        image_data = base64.b64decode(encoded_image)
        print('image_data: ', image_data)
        
        predicted_letters = preprocess_image(image_data)
        logger.debug("Predicted letters: %s", predicted_letters)
        print('predicted_letters: ', predicted_letters)
        
        predicted_word = ''.join(predicted_letters)
        logger.debug("Predicted word: %s", predicted_word)
        
        response = {
            'statusCode': 200,
            'body': json.dumps({'predicted_word': predicted_word})
        }

    except Exception as e:
        print("entrou no except")
        print(logger.error("Error: %s", str(e)))
        logger.error("Error: %s", str(e))
        response = {
            'statusCode': 500,
            'body': json.dumps({'error': 'An error occurred'})
        }

    return response


####################################################################################################################


    

