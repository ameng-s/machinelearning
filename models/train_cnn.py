
import os
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


# ==========================================
# 1. KONFIGURASI
# ==========================================

DATASET_PATH = "dataset/preprocessed"
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
# 3. DATA AUGMENTATION
# ==========================================

print("\n===== DATA AUGMENTATION =====")

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1)
], name="data_augmentation")


# ==========================================
# 4. MEMBANGUN ARSITEKTUR CNN
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
# 5. COMPILE MODEL
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
# 6. CALLBACKS
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
# 7. TRAINING MODEL
# ==========================================

print("\n===== MULAI TRAINING CNN =====")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ],
    shuffle=False
)


# ==========================================
# 8. LOAD MODEL TERBAIK
# ==========================================

print("\n===== MEMUAT MODEL TERBAIK =====")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model terbaik berhasil dimuat!")
print("Lokasi model:", MODEL_PATH)


# ==========================================
# 9. EVALUASI MODEL
# ==========================================

print("\n===== EVALUASI MODEL =====")

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")


# ==========================================
# 10. CLASSIFICATION REPORT
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
    "evaluation/cnn_classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)


# ==========================================
# 11. CONFUSION MATRIX
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

plt.title("CNN Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "evaluation/cnn_confusion_matrix.png",
    dpi=300
)

plt.close()


# ==========================================
# 12. GRAFIK ACCURACY
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

plt.title("CNN Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation/cnn_accuracy.png",
    dpi=300
)

plt.close()


# ==========================================
# 13. GRAFIK LOSS
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

plt.title("CNN Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "evaluation/cnn_loss.png",
    dpi=300
)

plt.close()


# ==========================================
# 14. SELESAI
# ==========================================

print("\n===== PROSES CNN SELESAI =====")

print("Model      : models/cnn_fish_freshness.keras")
print("Evaluation : evaluation/")
print(f"Accuracy   : {test_accuracy * 100:.2f}%")
print(f"Loss       : {test_loss:.4f}")

print("\nSemua proses berhasil dijalankan!")
