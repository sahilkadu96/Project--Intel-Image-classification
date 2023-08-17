from flask import Flask, session, redirect, render_template, request
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, FileField
from wtforms.validators import DataRequired
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] == 'MY_SECRET_KEY'

model = load_model(r'C:\Users\Sahil\Data science Machine Learning\INeuron FSDS\Deep Learning\Assignment5\intel_image\intel_image_cnn.h5', compile = False)

label = {0: 'buildings',
 1: 'forest',
 2: 'glacier',
 3: 'mountain',
 4: 'sea',
 5: 'street'}

upload_folder = r'C:\Users\Sahil\Data science Machine Learning\INeuron FSDS\Deep Learning\Assignment5\intel_image\static\uploads'
app.config['UPLOAD'] = upload_folder

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('home.html')


@app.route('/predict_class', methods = ['GET', 'POST'])
def predict_class():
    if request.method == 'POST':
        file = request.files['img']
        file_name = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], file_name))
        image = os.path.join(app.config['UPLOAD'], file_name)

        img = load_img(image, target_size = (150, 150))
        img_array = img_to_array(img)
        img_array = img_array/255
        img_array = np.expand_dims(img_array, axis = 0)
        classes = model.predict([img_array])
        class_index = np.argmax(classes, axis = 1)[0]
        class_name = label[class_index]

        return render_template('result.html', class_name = class_name, image = image)





if __name__ == '__main__':
    app.run()

