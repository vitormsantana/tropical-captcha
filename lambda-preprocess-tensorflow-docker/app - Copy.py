import cv2
import numpy as np
import os
import tempfile
import boto3
import base64
from PIL import Image
from io import BytesIO

def lambda_handler(event, context):
    try:
        s3_client = boto3.client('s3')
        input_bucket = event['Records'][0]['s3']['bucket']['name']
        input_key = event['Records'][0]['s3']['object']['key']
        output_bucket = "tropical-captcha-bucket"  # Replace with your S3 bucket name
        output_key = f"preprocessedImages/{os.path.basename(input_key)}"

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, os.path.basename(input_key))
            output_path = os.path.join(tmp_dir, f"preprocessed_{os.path.basename(input_key)}")

            # Download image from S3
            s3_client.download_file(input_bucket, input_key, input_path)

            # Decode and process base64 image
            processed_image = decode_base64_and_process(event['body']['data'])

            if processed_image is not None:
                # Save processed image to a temporary location
                cv2.imwrite(output_path, processed_image)

                # Upload processed image to S3
                s3_client.upload_file(output_path, output_bucket, output_key)

                return {
                    'statusCode': 200,
                    'body': f"Preprocessed image saved to: s3://{output_bucket}/{output_key}"
                }
            else:
                return {
                    'statusCode': 500,
                    'body': "Error decoding or processing the image"
                }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            'statusCode': 500,
            'body': f"Error: {e}"
        }

def generate_processed_images(image):
    # Process the image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply threshold
    threshold_value = 15
    thresholded = apply_threshold(gray_image, threshold_value)
    # Apply dilate and erode
    processed_image = apply_erode(apply_dilate(thresholded))
    return processed_image

def apply_threshold(image, threshold_value):
    thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)[1]
    return thresholded

def apply_dilate(image):
    dilated = cv2.dilate(image, np.ones((2, 3), np.uint8), iterations=1)
    return dilated

def apply_erode(image):
    eroded = cv2.erode(image, np.ones((3, 2), np.uint8), iterations=1)
    return eroded

def decode_base64_and_process(encoded_data):
    try:
        # Extract base64 data
        base64_data = encoded_data.split(',')[1]

        # Decode base64 to bytes
        img_data = base64.b64decode(base64_data)

        # Read image using PIL
        image = Image.open(BytesIO(img_data))

        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Apply processing to the image
        processed_image = generate_processed_images(image_np)

        return processed_image

    except Exception as e:
        print(f"An error occurred while decoding and processing: {e}")
        return None

