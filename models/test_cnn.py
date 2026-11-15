
import os
import numpy as np
import tensorflow as tf

# ==========================================
# 1. KONFIGURASI
# ==========================================

MODEL_PATH = "models/cnn_fish_freshness.keras"
IMAGE_SIZE = (128, 128)

CLASS_NAMES = [
    "Fresh",
    "Highly Fresh",
    "Not Fresh"
]

# Ganti dengan lokasi gambar yang ingin diuji
IMAGE_PATH = "dataset/preprocessed/test/Fresh"

# ==========================================
# 2. LOAD MODEL
# ==========================================

print("\n===== MEMUAT MODEL CNN =====")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model berhasil dimuat!")

# ==========================================
# 3. MEMILIH GAMBAR
# ==========================================

print("\n===== MEMUAT GAMBAR =====")

if os.path.isdir(IMAGE_PATH):
    files = [
        f for f in os.listdir(IMAGE_PATH)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not files:
        raise ValueError("Tidak ada gambar di folder tersebut.")

    image_path = os.path.join(IMAGE_PATH, files[0])
else:
    image_path = IMAGE_PATH

# ==========================================
# 4. PREPROCESSING GAMBAR
# ==========================================

img = tf.keras.utils.load_img(
    image_path,
    target_size=IMAGE_SIZE
)

img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# ==========================================
# 5. PREDIKSI
# ==========================================

print("\n===== MELAKUKAN PREDIKSI =====")

prediction = model.predict(img_array)

predicted_index = np.argmax(prediction[0])
predicted_class = CLASS_NAMES[predicted_index]
confidence = prediction[0][predicted_index] * 100

print("\n===== HASIL PREDIKSI =====")
print("Nama gambar :", os.path.basename(image_path))
print("Hasil       :", predicted_class)
print(f"Confidence  : {confidence:.2f}%")
