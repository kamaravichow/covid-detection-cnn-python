from keras.models import load_model
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

image_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


st.title("Covid-19 Detection")

st.write("This is a simple image classification web app to predict whether the patient is suffering from Covid-19 or not")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    # resize the image to 224x224
    image = image.resize((320, 320))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    # save the image to the current directory
    image.save('image.jpg')
    st.write("Classifying...")
    
    model = load_model('model\mv1-cov19xray.h5', custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
    
    img = load_img('image.jpg', target_size=(320, 320))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = image_generator.standardize(x)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    if classes[0]>0.5:
        st.write("Covid-19 Positive")
    else:
        st.write("Covid-19 Negative")