import os
import time
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

TRAIN_DIR = "dataset/preprocessed/train"
TEST_DIR = "dataset/preprocessed/test"
EVAL_OUT_DIR = "evaluation"

CLASSES = [
    "Highly Fresh",
    "Fresh",
    "Not Fresh"
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
PROPERTIES = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
DISTANCES = [1, 3, 5]

def extract_features_from_image(image_path):
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        return None
    
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    
    glcm = graycomatrix(
        gray_img,
        distances=DISTANCES,
        angles=ANGLES,
        levels=256,
        symmetric=True,
        normed=True
    )
    
    features = []
    for prop in PROPERTIES:
        prop_values = graycoprops(glcm, prop).flatten()
        features.extend(prop_values)
        
    for channel in range(3):
        features.append(np.mean(rgb_img[:, :, channel]))
        features.append(np.std(rgb_img[:, :, channel]))
        features.append(np.mean(hsv_img[:, :, channel]))
        features.append(np.std(hsv_img[:, :, channel]))
        
    return np.array(features, dtype=np.float64)

def load_dataset_features(base_dir):
    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {base_dir}. Jalankan split_dataset.py dan preprocess_dataset.py terlebih dahulu."
        )
        
    X_list = []
    y_list = []
    
    for label_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(base_dir, class_name)
        if not os.path.exists(class_folder):
            continue
            
        files = [
            f for f in sorted(os.listdir(class_folder))
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
        
        for file_name in files:
            file_path = os.path.join(class_folder, file_name)
            feat = extract_features_from_image(file_path)
            if feat is not None:
                X_list.append(feat)
                y_list.append(label_idx)
                
    return np.array(X_list), np.array(y_list)

if __name__ == "__main__":
    print("\n--MEMULAI EKSTRAKSI FITUR (GLCM + WARNA) UNTUK KNN--")
    start_feat_time = time.time()
    
    X_train, y_train = load_dataset_features(TRAIN_DIR)
    X_test, y_test = load_dataset_features(TEST_DIR)
    
    feat_time = time.time() - start_feat_time
    print(f"Total data training : {len(X_train)} sampel")
    print(f"Total data testing  : {len(X_test)} sampel")
    print(f"Jumlah fitur/gambar : {X_train.shape[1]} fitur")
    print(f"Waktu ekstraksi     : {feat_time:.4f} detik")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n-- MELATIH MODEL KNN --")
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    start_train_time = time.time()
    
    grid_search = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_knn = grid_search.best_estimator_
    train_time = time.time() - start_train_time
    
    y_pred = best_knn.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    print("\n-- HASIL EVALUASI MODEL KNN --")
    print(f"Parameter Terbaik : {grid_search.best_params_}")
    print(f"Waktu Pelatihan   : {train_time:.4f} detik")
    print(f"Accuracy          : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (Macro) : {prec:.4f}")
    print(f"Recall (Macro)    : {rec:.4f}")
    print(f"F1-Score (Macro)  : {f1:.4f}")
    print("\n----- Classification Report -----")
    print(classification_report(y_test, y_pred, target_names=CLASSES, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    print("----- Confusion Matrix -----")
    print(cm)
    
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)
    model_path = os.path.join(EVAL_OUT_DIR, "knn_model.pkl")
    scaler_path = os.path.join(EVAL_OUT_DIR, "knn_scaler.pkl")
    cm_path = os.path.join(EVAL_OUT_DIR, "confusion_matrix_knn.png")
    
    joblib.dump(best_knn, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel KNN berhasil disimpan di : {model_path}")
    print(f"Scaler KNN berhasil disimpan di: {scaler_path}")
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Confusion Matrix - KNN Fish Eye Freshness")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    print(f"Confusion Matrix disimpan di   : {cm_path}")
    plt.show()