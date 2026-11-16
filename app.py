
import os
import numpy as np
import tensorflow as tf

from io import BytesIO
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Konfigurasi model CNN + Class Weight
MODEL_PATH = "models/cnn_fish_freshness.keras"
IMAGE_SIZE = (128, 128)

# Urutan kelas harus sama dengan saat training CNN
CLASS_NAMES = ["Fresh", "Highly Fresh", "Not Fresh"]

# Memuat model CNN + Class Weight
model = tf.keras.models.load_model(MODEL_PATH)


# Halaman utama website
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# API untuk prediksi gambar ikan
@app.route("/predict", methods=["POST"])
def predict():

    # Memeriksa apakah gambar dikirim
    if "file" not in request.files:
        return jsonify({
            "error": "Gambar belum diunggah"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Silakan pilih gambar terlebih dahulu"
        }), 400

    try:
        # Membaca gambar dan mengubah ukurannya
        img = tf.keras.utils.load_img(
            BytesIO(file.read()),
            target_size=IMAGE_SIZE,
            color_mode="rgb"
        )

        # Mengubah gambar menjadi array
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Melakukan prediksi menggunakan CNN
        prediction = model.predict(img_array, verbose=0)

        # Mengambil kelas dengan nilai tertinggi
        predicted_index = np.argmax(prediction[0])
        predicted_class = CLASS_NAMES[predicted_index]

        # Menghitung confidence
        confidence = float(
            prediction[0][predicted_index] * 100
        )

        # Mengirim hasil prediksi
        return jsonify({
            "prediction": predicted_class,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# Menjalankan Flask
import os
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
