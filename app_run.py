from flask import Flask, render_template, request
from minst_model import ModelLoader
from PIL import Image
from binascii import a2b_base64
import numpy as np
import base64
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import tensorflow.keras.models
import re
import sys

import os

json_architecture = os.getcwd() + '\models\\model.json'
model_weights = os.getcwd() + '\models\\model.h5'
model = ModelLoader(json_architecture, model_weights)
app = Flask(__name__)




@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imageData = request.get_data()

    head, imdata = imageData.split(b',')
    data = base64.b64decode(imdata)

    with open('image_output.png', 'wb') as image_file:
        image_file.write(data)
    image_ = Image.open('image_output.png')
    image_ = image_.resize((28, 28)).convert('L')
    image_ = np.invert(image_)
    image_ = image_.reshape(1, 28, 28, 1)
    print(image_.shape)
    predicted_digit = model.predict(image_)
    response = predicted_digit
    return response


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='127.0.0.1', port=port)
