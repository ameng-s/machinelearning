
import os
import numpy as np
import tensorflow as tf
import streamlit as st
from io import BytesIO

# =========================
# PAGE CONFIGURATION
# =========================
st.set_page_config(
    page_title="FishFresh - Fish Freshness Classification",
    page_icon="🐟",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    * {
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    .stApp {
        background-color: #f3f6fa;
        color: #263445;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .navbar {
        background: white;
        padding: 18px 25px;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }

    .navbar h2 {
        color: #168b83;
        margin: 0;
        font-size: 24px;
    }

    .navbar span {
        color: #718096;
        font-size: 14px;
    }

    .hero {
        text-align: center;
        padding: 15px 10px 25px;
    }

    .hero h1 {
        color: #263445;
        font-size: 32px;
    }

    .hero p {
        color: #718096;
        font-size: 15px;
    }

    .upload-card {
        background: white;
        padding: 28px;
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.06);
        margin-top: 10px;
    }

    .upload-card h3 {
        color: #263445;
        text-align: center;
    }

    .upload-card p {
        color: #718096;
        text-align: center;
    }

    [data-testid="stFileUploader"] {
        background: #f6fbfa;
        border: 2px dashed #9acdc7;
        border-radius: 12px;
        padding: 15px;
    }

    .stButton > button {
        background: #168b83;
        color: white;
        border: none;
        border-radius: 10px;
        width: 100%;
        font-weight: 600;
        padding: 12px;
    }

    .stButton > button:hover {
        background: #116e68;
        color: white;
        border: none;
    }

    .result-box {
        padding: 18px;
        border-radius: 10px;
        margin-top: 20px;
        border-left: 5px solid;
    }

    .fresh {
        background: #fffbeb;
        border-color: #d97706;
    }

    .highly-fresh {
        background: #ecfdf5;
        border-color: #059669;
    }

    .not-fresh {
        background: #fef2f2;
        border-color: #dc2626;
    }

    .result-box h3 {
        margin-top: 0;
    }

    .footer {
        text-align: center;
        color: #718096;
        font-size: 13px;
        padding: 25px 0;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# MODEL CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "cnn_fish_freshness.keras"
)

IMAGE_SIZE = (128, 128)

CLASS_NAMES = ["Fresh", "Highly Fresh", "Not Fresh"]


# =========================
# LOAD CNN MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load CNN model: {e}")
    st.stop()


# =========================
# NAVBAR
# =========================
st.markdown("""
<div class="navbar">
    <h2>FishFresh</h2>
    <span>Fish Freshness Detection</span>
</div>
""", unsafe_allow_html=True)


# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">
    <h1>Fish Freshness Classification</h1>
    <p>
        Upload a fish eye image and let our AI model
        classify its freshness level.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# UPLOAD SECTION
# =========================
st.markdown("""
<div class="upload-card">
    <h3>Upload Fish Image</h3>
    <p>Select an image of a fish eye to start the classification.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📷 Choose an Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# =========================
# IMAGE PREVIEW
# =========================
if uploaded_file is not None:

    st.image(
        uploaded_file,
        caption="Image Preview",
        use_container_width=True
    )

    if st.button("Predict Freshness", use_container_width=True):

        with st.spinner("Analyzing image, please wait..."):

            try:
                img = tf.keras.utils.load_img(
                    BytesIO(uploaded_file.getvalue()),
                    target_size=IMAGE_SIZE,
                    color_mode="rgb"
                )

                img_array = tf.keras.utils.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                predicted_index = np.argmax(prediction[0])
                predicted_class = CLASS_NAMES[predicted_index]

                confidence = float(
                    prediction[0][predicted_index] * 100
                )

                if predicted_class == "Fresh":
                    status_class = "fresh"
                elif predicted_class == "Highly Fresh":
                    status_class = "highly-fresh"
                else:
                    status_class = "not-fresh"

                st.markdown(f"""
                <div class="result-box {status_class}">
                    <h3>Prediction Result</h3>
                    <p>
                        Freshness Level:
                        <strong>{predicted_class}</strong>
                    </p>
                    <p>
                        Confidence:
                        <strong>{confidence:.2f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction error: {e}")


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
    FishFresh - Fish Freshness Classification System
</div>
""", unsafe_allow_html=True)
