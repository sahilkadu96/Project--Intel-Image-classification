import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import streamlit as st
from PIL import Image


model = load_model(r'C:\Users\Sahil\.spyder-py3\intel_image\intel_image_cnn.h5', compile = False)


st.title('Intel image classification')

label = {0: 'buildings',
 1: 'forest',
 2: 'glacier',
 3: 'mountain',
 4: 'sea',
 5: 'street'}

image = st.file_uploader('Upload image', type = ['jpg'])

if st.button('Predict'):
  img = load_img(image, target_size = (150, 150))
  img_array = img_to_array(img)
  img_array = img_array/255
  img_array = np.expand_dims(img_array, axis = 0)

  classes = model.predict([img_array])
  class_index = np.argmax(classes, axis = 1)[0]
  class_name = label[class_index]

  st.image(image)

  st.write(f'Predicted class is {class_name}')