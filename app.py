import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
import os

# Check if the model file exists
keras_model_path = "./Image_classify.keras"
if not os.path.isfile(keras_model_path):
    st.write(f"Model file {keras_model_path} does not exist.")
    raise SystemExit

# Try to load the model
try:
    model = load_model(keras_model_path)
except Exception as e:
    st.write(f"Error loading model: {e}")
    raise SystemExit

# Your categories and image dimensions
data_cat = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
img_height = 180
img_width = 180

# Get the image name from the user
image = st.text_input('Enter Image name', 'Apple.jpg')

# Check if the image file exists
if not os.path.isfile(image):
    st.write(f"Image file {image} does not exist.")
    raise SystemExit

# Load and preprocess the image
image_load = tf.keras.utils.load_img(image, target_size=(img_height, img_width))
img_arr = tf.keras.utils.img_to_array(image_load)
img_bat = tf.expand_dims(img_arr, 0)

# Make a prediction
predict = model.predict(img_bat)
score = tf.nn.softmax(predict[0])

# Display the results
st.image(image, width=200)
st.write('Veg/Fruit in image is ' + data_cat[np.argmax(score)])
st.write('With accuracy of ' + str(100 * np.max(score)) + '%')