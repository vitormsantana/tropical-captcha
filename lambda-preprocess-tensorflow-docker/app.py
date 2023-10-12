import cv2
import numpy as np
import os
import tempfile
import boto3
import base64
from PIL import Image
from io import BytesIO


def lambda_handler(event, context):
    
    s3_client = boto3.client('s3')
    input_bucket = event['Records'][0]['s3']['bucket']['name']
    input_key = event['Records'][0]['s3']['object']['key']
    output_bucket = "tropical-captcha-bucket"  # Replace with your S3 bucket name
    output_key = f"preprocessedImages/{os.path.basename(input_key)}"

    return input_bucket