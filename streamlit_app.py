import numpy as np
import streamlit as st
import tensorflow as tf
from io import BytesIO
from PIL import Image

MODEL_PATH = "models/cnn_fish_freshness.keras"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Fresh", "Highly Fresh", "Not Fresh"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("FishFresh - Klasifikasi Kesegaran Ikan")

file = st.file_uploader("Upload gambar ikan", type=["jpg", "jpeg", "png"])

if file:
    data = file.read()
    st.image(Image.open(BytesIO(data)), use_container_width=True)

    img = tf.keras.utils.load_img(
        BytesIO(data),
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    idx = int(np.argmax(prediction[0]))
    confidence = float(prediction[0][idx] * 100)

    st.subheader(f"Hasil: {CLASS_NAMES[idx]}")
    st.write(f"Confidence: {confidence:.2f}%")