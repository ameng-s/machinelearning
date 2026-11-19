
import streamlit as st
import numpy as np
import tensorflow as tf

from io import BytesIO


# =========================
# 1. KONFIGURASI
# =========================

st.set_page_config(
    page_title="FishFresh",
    page_icon="🐟",
    layout="centered"
)

MODEL_PATH = "models/cnn_fish_freshness.keras"

IMAGE_SIZE = (128, 128)

CLASS_NAMES = ["Fresh", "Not Fresh"]


# =========================
# 2. LOAD CSS
# =========================

with open("static/css/style.css", "r", encoding="utf-8") as file:
    css = file.read()

st.markdown(
    f"<style>{css}</style>",
    unsafe_allow_html=True
)

# CSS tambahan untuk Streamlit
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f3f6fa;
    color: #263445;
}

/* Hilangkan header bawaan */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Navbar */
.navbar-custom {
    background: white;
    padding: 18px 25px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.navbar-custom h2 {
    color: #168b83;
    font-size: 24px;
    margin: 0;
}

.navbar-custom span {
    color: #718096;
    font-size: 14px;
}

/* Hero */
.hero-custom {
    text-align: center;
    padding: 30px 10px 20px;
}

.hero-custom h1 {
    color: #263445;
    font-size: 32px;
}

.hero-custom p {
    color: #718096;
    font-size: 14px;
}

/* Upload */
[data-testid="stFileUploader"] {
    background: #f6fbfa;
    border: 2px dashed #9acdc7;
    border-radius: 12px;
    padding: 15px;
}

/* Button */
.stButton button {
    width: 100%;
    background: #168b83;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px;
    font-weight: 600;
}

.stButton button:hover {
    background: #116e68;
    color: white;
}

/* Result */
.result-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin-top: 15px;
    border-left: 5px solid #168b83;
}

.footer-custom {
    text-align: center;
    color: #718096;
    font-size: 13px;
    padding: 30px 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# 3. LOAD MODEL CNN
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


# =========================
# 4. NAVBAR
# =========================

st.markdown("""
<div class="navbar-custom">
    <h2>FishFresh</h2>
    <span>Fish Freshness Detection</span>
</div>
""", unsafe_allow_html=True)


# =========================
# 5. HERO
# =========================

st.markdown("""
<div class="hero-custom">
    <h1>Fish Freshness Classification</h1>
    <p>
        Upload a fish eye image and let our AI model
        classify its freshness level.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# 6. UPLOAD IMAGE
# =========================

st.markdown(
    "<h3 style='text-align:center;'>Upload Fish Image</h3>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;color:#718096;'>"
    "Select an image of a fish eye to start the classification."
    "</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"]
)


# =========================
# 7. PREVIEW IMAGE
# =========================

if uploaded_file is not None:

    image_bytes = uploaded_file.getvalue()

    st.image(
        image_bytes,
        caption="Image Preview",
        use_container_width=True
    )

    # =========================
    # 8. PREDICTION
    # =========================

    if st.button("Predict Freshness"):

        with st.spinner("Analyzing image, please wait..."):

            try:

                img = tf.keras.utils.load_img(
                    BytesIO(image_bytes),
                    target_size=IMAGE_SIZE,
                    color_mode="rgb"
                )

                img_array = tf.keras.utils.img_to_array(img)

                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                idx = int(np.argmax(prediction[0]))

                confidence = float(
                    prediction[0][idx] * 100
                )

                result = CLASS_NAMES[idx]

                # =========================
                # 9. DISPLAY RESULT
                # =========================

                st.markdown(
                    f"""
                    <div class="result-card">
                        <h3 style="color:#168b83;">
                            Prediction Result
                        </h3>

                        <p>Freshness Level:
                            <strong>{result}</strong>
                        </p>

                        <p>Confidence:
                            <strong>{confidence:.2f}%</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            except Exception as e:

                st.error(f"Prediction Error: {e}")

else:

    st.info("Please select an image first.")


# =========================
# 10. FOOTER
# =========================

st.markdown("""
<div class="footer-custom">
    FishFresh - Fish Freshness Classification System
</div>
""", unsafe_allow_html=True)
