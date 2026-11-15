
import os
import cv2 
import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

BASE = "dataset/preprocessed"
CLASSES = ["Highly Fresh", "Fresh", "Not Fresh"]
IMG_SIZE = (128, 128)


def load_split(split):
    X, y = [], []
    for label, kelas in enumerate(CLASSES):
        folder = os.path.join(BASE, split, kelas)
        for nama in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, nama), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            X.append(hog(img, orientations=9, pixels_per_cell=(16, 16),
            cells_per_block=(2, 2)))
            y.append(label)
    return np.array(X), np.array(y)


print("Memuat Data")

# Menghitung waktu ekstraksi fitur
start_extract = time.perf_counter()

X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

end_extract = time.perf_counter()
extraction_time = end_extract - start_extract

print("Train:", X_train.shape, "| Test:", X_test.shape)
print(f"Waktu Ekstraksi Fitur: {extraction_time:.2f} detik")

print("Training SVM...")

model = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")

# Menghitung waktu training
start_train = time.perf_counter()

model.fit(X_train, y_train)

end_train = time.perf_counter()
training_time = end_train - start_train

print(f"Waktu Training: {training_time:.2f} detik")

pred = model.predict(X_test)
print(classification_report(y_test, pred, target_names=CLASSES))

os.makedirs("evaluation", exist_ok=True)
ConfusionMatrixDisplay.from_predictions(
    y_test, pred, display_labels=CLASSES, cmap="Blues"
)
plt.title("Confusion Matrix SVM")
plt.savefig("evaluation/svm_confusion_matrix.png", bbox_inches="tight")

joblib.dump(model, "models/svm_model.pkl")
