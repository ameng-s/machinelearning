
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# 1. KONFIGURASI
# ==========================================

DATASET_PATH = "dataset/preprocessed"

# Menggunakan nama berbeda agar model lama aman
MODEL_PATH = "models/cnn_fish_freshness.keras"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42

CLASS_NAMES = [
    "Fresh",
    "Highly Fresh",
    "Not Fresh"
]

os.makedirs("models", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\n===== MEMUAT DATASET =====")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int"
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nUrutan kelas:", train_dataset.class_names)

assert train_dataset.class_names == CLASS_NAMES, (
    "Urutan kelas tidak sesuai!"
)

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================
# 3. MENGHITUNG CLASS WEIGHT
# ==========================================

print("\n===== MENGHITUNG CLASS WEIGHT =====")

train_labels = np.concatenate([
    labels.numpy()
    for images, labels in train_dataset
])

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weight_dict = {
    int(class_index): float(weight)
    for class_index, weight in zip(
        np.unique(train_labels),
        class_weights
    )
}

print("\nClass Weight:")

for class_index, weight in class_weight_dict.items():
    print(f"{CLASS_NAMES[class_index]}: {weight:.4f}")


# ==========================================
# 4. OPTIMASI DATASET
# ==========================================

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

print("\n===== DATA AUGMENTATION =====")

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1)
], name="data_augmentation")


# ==========================================
# 6. MEMBANGUN ARSITEKTUR CNN
# ==========================================

print("\n===== MEMBANGUN MODEL CNN =====")

model = models.Sequential([

    layers.Input(shape=(128, 128, 3)),

    data_augmentation,

    layers.Rescaling(1.0 / 255),

    # Convolution Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Convolution Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        3,
        activation="softmax"
    )

])

model.summary()


# ==========================================
# 7. COMPILE MODEL
# ==========================================

print("\n===== COMPILE MODEL =====")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 8. CALLBACKS
# ==========================================

print("\n===== MENYIAPKAN CALLBACKS =====")

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ==========================================
# 9. TRAINING MODEL
# ==========================================

print("\n===== MULAI TRAINING CNN + CLASS WEIGHT =====")

start_train = time.perf_counter()

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ],
    class_weight=class_weight_dict,
    shuffle=False
)

end_train = time.perf_counter()
training_time = end_train - start_train

print(f"\nWaktu Training CNN: {training_time:.2f} detik")
print(f"Jumlah Epoch: {len(history.history['loss'])}")


# ==========================================
# 10. LOAD MODEL TERBAIK
# ==========================================

print("\n===== MEMUAT MODEL TERBAIK =====")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model terbaik berhasil dimuat!")
print("Lokasi model:", MODEL_PATH)


# ==========================================
# 11. EVALUASI MODEL
# ==========================================

print("\n===== EVALUASI MODEL =====")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ==========================================
# 12. CLASSIFICATION REPORT
# ==========================================

print("\n===== CLASSIFICATION REPORT =====")

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
])

y_pred_prob = model.predict(test_dataset)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)

with open(
    "evaluation/cnn_weighted_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ==========================================
# 13. CONFUSION MATRIX
# ==========================================

print("\n===== CONFUSION MATRIX =====")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("CNN + Class Weight Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "evaluation/cnn_weighted_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 14. GRAFIK ACCURACY
# ==========================================

print("\n===== MEMBUAT GRAFIK ACCURACY =====")

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("CNN + Class Weight Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation/cnn_weighted_accuracy.png",
    dpi=300
)

plt.close()


# ==========================================
# 15. GRAFIK LOSS
# ==========================================

print("\n===== MEMBUAT GRAFIK LOSS =====")

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("CNN + Class Weight Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation/cnn_weighted_loss.png",
    dpi=300
)

plt.close()


# ==========================================
# 16. SELESAI
# ==========================================

print("\n===== PROSES CNN + CLASS WEIGHT SELESAI =====")

print("Model      : models/cnn_fish_freshness_weighted.keras")
print("Evaluation : evaluation/")
print(f"Accuracy   : {test_accuracy * 100:.2f}%")
print(f"Loss       : {test_loss:.4f}")
print(f"Training Time : {training_time:.2f} detik")

print("\nSemua proses berhasil dijalankan!")
