
import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "models/cnn_fish_freshness.keras"
TEST_DIR = "dataset/preprocessed/test"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

CLASS_NAMES = ["Fresh", "Highly Fresh", "Not Fresh"]

print("Memuat model CNN...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Memuat data testing...")
test_data = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Mengambil prediksi seluruh gambar testing
predictions = model.predict(test_data, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

# Mengambil nama file dan label aslinya
file_paths = test_data.file_paths

print("\n===== GAMBAR NOT FRESH YANG SALAH DIPREDIKSI =====")

total_wrong = 0

for i, file_path in enumerate(file_paths):
    true_label = os.path.basename(os.path.dirname(file_path))
    predicted_label = CLASS_NAMES[predicted_labels[i]]

    if true_label == "Not Fresh" and predicted_label != "Not Fresh":
        confidence = predictions[i][predicted_labels[i]] * 100

        print("\nNama gambar :", os.path.basename(file_path))
        print("Label asli  :", true_label)
        print("Prediksi    :", predicted_label)
        print(f"Confidence  : {confidence:.2f}%")

        total_wrong += 1

print("\n==============================================")
print("Total gambar Not Fresh yang salah:", total_wrong)
