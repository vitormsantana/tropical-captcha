import tensorflow as tf
import base64
import json
import cv2
import numpy as np
import boto3
import pickle

# Initialize the S3 client
s3 = boto3.client('s3')

bucketName = 'tropical-captcha-bucket'

# Load your trained Keras model
MODEL_FILENAME = 'captcha_model_SimpleCNN.h5'
s3.download_file('tropical-captcha-bucket/tropical-captcha/LambdaFiles/', MODEL_FILENAME, '/tmp/' + MODEL_FILENAME)
model = tf.keras.models.load_model('/tmp/' + MODEL_FILENAME)

# Load the label binarizer from S3
MODEL_LABELS_FILENAME = 'model_labels.pkl'
s3.download_file('tropical-captcha-bucket/tropical-captcha/LambdaFiles/', MODEL_LABELS_FILENAME, '/tmp/' + MODEL_LABELS_FILENAME)
with open('/tmp/' + MODEL_LABELS_FILENAME, 'rb') as f:
    lb = pickle.load(f)

# Division positions
division_positions = [(5, 42), (37, 71), (65, 105), (94, 133), (125, 159), (155, 179)]

def preprocess_image(image_data):
    # Decode base64-encoded image
    decoded_image = base64.b64decode(image_data)
    nparr = np.frombuffer(decoded_image, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    thresholded_image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)[1]
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
        
    return predicted_letters
 

#def get_predicted_class(prediction):
#    # Process the prediction and get the predicted class
#    predicted_label_index = np.argmax(prediction, axis=1)
#    predicted_label = lb.classes_[predicted_label_index][0]
#    return predicted_label

def lambda_handler(event, context):
    body = json.loads(event['body'])
    encoded_image = body['image']

    predicted_letters = preprocess_image(encoded_image)
    predicted_word = ''.join(predicted_letters)  # Join the predicted letters into a string
    
    response = {
        'statusCode': 200,
        'body': json.dumps({'predicted_word': predicted_word})
    }

    return response

    return response