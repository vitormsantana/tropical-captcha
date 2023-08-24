import tensorflow as tf
import base64
import json
import cv2
import numpy as np
#import boto3
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow GPU-related logs

# Load the model
model_path = r"C:\Users\visantana\Documents\tropical-captcha\captcha_model_SimpleCNN.h5"
model = tf.keras.models.load_model(model_path, compile=False)

# Load the label binarizer
MODEL_LABELS_FILENAME = r"C:\Users\visantana\Documents\tropical-captcha\model_labels.pkl"
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

def cut_bottom(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[:-cut_height, :]

def cut_top(image, cut_percent):
    height = image.shape[0]
    cut_height = int(height * cut_percent)
    return image[cut_height:, :]

# Initialize the S3 client
#s3 = boto3.client('s3')

#bucketName = 'tropical-captcha-bucket'

# Load your trained Keras model
#MODEL_FILENAME = 'captcha_model_SimpleCNN.h5'
#s3.download_file('tropical-captcha-bucket/tropical-captcha/LambdaFiles/', MODEL_FILENAME, '/tmp/' + MODEL_FILENAME)
#model = tf.keras.models.load_model('/tmp/' + MODEL_FILENAME)

# Load the label binarizer from S3
#MODEL_LABELS_FILENAME = 'model_labels.pkl'
#s3.download_file('tropical-captcha-bucket/tropical-captcha/LambdaFiles/', MODEL_LABELS_FILENAME, '/tmp/' + MODEL_LABELS_FILENAME)
#with open('/tmp/' + MODEL_LABELS_FILENAME, 'rb') as f:
#    lb = pickle.load(f)

# Division positions
division_positions = [(5, 42), (37, 71), (65, 105), (94, 133), (125, 159), (155, 179)]

def preprocess_image(image_data):
    
    # Accessing the nested 'body' key within the event dictionary
    nested_body_string = event['body']['body']
    
    # Parse the JSON string contained within the 'body' key
    nested_body = json.loads(nested_body_string)
    
    print('nested_body: ', str(nested_body)[:66])

    # Extract the 'data' key from the nested 'body' dictionary
    image_data = nested_body['data']
    print('image_data: ', str(image_data)[:55])
    
    encoded_image_data = str(image_data).split(",")[1]
    decoded_image = base64.b64decode(encoded_image_data)
    
    # Check and add padding if needed
    #padding = len(image_data) % 4
    #if padding > 0:
    #    image_data += '=' * (4 - padding)
    
    # Decode the base64 image data
    #decoded_image = base64.b64decode(image_data)
        
    
    # Decode the base64 image data
    #decoded_image = base64.b64decode(image_data)
        
    print("Entering preprocess_image function")
    
    # Decode base64-encoded image
    #decoded_image = base64.b64decode(image_data)
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
        
    print("Predicted letters:", predicted_letters)
    return predicted_letters
 

def get_predicted_class(prediction):
    # Process the prediction and get the predicted class
    predicted_label_index = np.argmax(prediction, axis=1)
    predicted_label = lb.classes_[predicted_label_index][0]
    return predicted_label

def lambda_handler(event, context):
    print("Received event:", str(event)[:62])
    body = json.dumps(event['body'])
    #encoded_image = body['image']
    

    
    print("event: ", str(event)[:62])
    predicted_letters = preprocess_image(event)
    #predicted_letters = preprocess_image(encoded_image)
    print("Predicted letters:", predicted_letters)
    predicted_word = ''.join(predicted_letters)  # Join the predicted letters into a string
    print("Predicted word:", predicted_word)
    response = {
        'statusCode': 200,
        'body': json.dumps({'predicted_word': predicted_word})
    }

    return response

encoded_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAALQAAAAyCAYAAAD1JPH3AAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAABhCSURBVHhe3Z1ZyF1FEseD+x6JRo0R933DLSpxjfsa9x133HA3ooKKW9Sog0aMiBsOIcQgLgOOiiI46qDoDBjNg8o8RJ8iDEPmzXk7c3893/9at251nz7fd+M484fiyz2nl+rqf1VX9zn3ZtLUqVP/OmnSpCYna6+99r823njjlVOmTPnHtGnTVsycOXPp7Nmz/3zOOef8ycrDDz+85Pnnn//9RGTOnDl/8O0is2bN+suOO+64XNLT+e+Rrl1Ebc2YMWOZ+tEYnnnmmYWfffbZ/B9//HFe0zT318rKlSvnlurQ5lNPPbWIvnbbbbe/RePYcsstV1x//fV/pK2oDSvoWlNuVMLY6HPZsmW/05x1tVEktPH4448vZizY6IMPPlgQlauRSZMnT/6nN+r/qmywwQbNPvvsk/59wAEHNJdeemlz1VVXNT0SJXnjjTeajz76aEB+/vnnnh0G8corrzTnnXdec999941dqcNbb73VqQ79oOvRRx/d7LrrrgNj6QWR1FZvksdKD0J1GR/jyJXjenSPcefqRPjqq6+ahQsXJp2WLFmS/iJcnygeeOCBNJabbrqpP35sWYvPP/987F9NM6kn97/33nvNCSec0G8MWWuttZq99tqrOf7444lmzbbbbtusscYaA2VGKUxgJFtvvXXqW7L99tsn0nI9ameiArEOPvjgZocddmiOOOKI5uabb24WLFgwYLQcIFkXQkOoa665pnnzzTfTZybR67POOus0t9xyS7NixYpURqBuL5In/UrE4np0L3K+5cuXN99+++3Yp18AP6QPYxw1GAu69CJ+v5/nnntu7G4ZL7744oBeidDpXz0wmMsuuywZUQ0jXMMrNakooAhHx3fffXdz1llnNYcddlhz6KGHNkceeWSSLbbYYqCd/weB8Dg/E/Dqq68OEADSdYlYniiW0L20ZMB+zMmjjz46VvM/qHGyHLzzobf68tGRaM4KR/+5qJ5zhhJoy7eHXnAqWjk9rP0uuOCCdG2A0AITw2AjQt55551jpX4BA6E88sknn4SDRkE5QSQysITIpT61/CNRXQRyPfvss/1I1sXATKbawUjq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"
# Convert to JSON format
image_json = {'data': encoded_image}
encoded_image_json = json.dumps(image_json)
# Now you can use encoded_image_json in your lambda_handler
body = {'body': encoded_image_json}
event = {'body': body}
print("Encoded image:", encoded_image[:62])
lambda_handler(event, None)