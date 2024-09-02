import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import warnings

warnings.filterwarnings("ignore")


image_classifier = tf.keras.models.load_model("image_classification.keras")
import streamlit as st
st.title("Image Classification Model using Basic CNN Architecture")


def prediction(input_image):
    test_image = image.load_img(input_image, target_size=(65,65))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = image_classifier.predict(test_image)
    classes = ["kintu", "krishna"]
    op = classes[np.argmax(result)]
    return op

input_image = st.camera_input("Camera Input Image")
if input_image is not None:
    result = prediction(input_image)
    if result == "kintu":
        st.success("Kintu")
        st.image(input_image)
    else:
        st.warning("Kintu")
