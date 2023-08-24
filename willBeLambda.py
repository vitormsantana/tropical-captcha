import tensorflow as tf
import base64
import cv2
import numpy as np
import boto3
import pickle
import os

# Initialize the S3 client
s3 = boto3.client('s3')

# Specify the bucket name and model file name
bucketName = 'tropical-captcha-bucket'
MODEL_FILENAME = 'captcha_model_SimpleCNN.h5'
s3_key = 'tropical-captcha/LambdaFiles/' + MODEL_FILENAME

# Download the model file from S3 to the Lambda /tmp directory
local_model_path = '/tmp/' + MODEL_FILENAME
s3.download_file(bucketName, s3_key, local_model_path)

# Load the Keras model from the downloaded file
model = tf.keras.models.load_model(local_model_path)

# Specify the bucket name and label binarizer file name
MODEL_LABELS_FILENAME = 'model_labels.pkl'
s3_key_labels = 'tropical-captcha/LambdaFiles/' + MODEL_LABELS_FILENAME

# Download the label binarizer file from S3 to the Lambda /tmp directory
local_labels_path = '/tmp/' + MODEL_LABELS_FILENAME
s3.download_file(bucketName, s3_key_labels, local_labels_path)

# Load the label binarizer
with open(local_labels_path, "rb") as f:
    lb = pickle.load(f)

# Division positions
division_positions = [(5, 42), (37, 71), (65, 105), (94, 133), (125, 159), (155, 179)]

def cut_bottom(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[:-cut_height, :]

def cut_top(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[cut_height:, :]

def lambda_handler(event, context):
    print("Received event:", event)
    encoded_image = event['encoded_image']

    # Preprocess the image
    processed_image = preprocess_image(encoded_image)
    print('predicted_letters: ', predicted_letters)

    predicted_word = ''.join(predicted_letters)

    response = {
        'statusCode': 200,
        'body': json.dumps({'predicted_word': predicted_word})
    }
    print(response)
except Exception as e:
    print("Error:", e)
    response = {
        'statusCode': 500,
        'body': json.dumps({'error': str(e)})
    }