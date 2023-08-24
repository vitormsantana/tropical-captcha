import base64

image_path = "C:/Users/visantana/Documents/tropical-captcha/Captchas/captcha_AlreadyProcessed/captcha_33.png"  
with open(image_path, "rb") as image_file:
    base64_encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

print(base64_encoded_image)
