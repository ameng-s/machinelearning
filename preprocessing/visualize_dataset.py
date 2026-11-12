
import os
import random
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. KONFIGURASI
# ==========================================

DATASET_PATH = "dataset/preprocessed/train"

CLASSES = [
    "Highly Fresh",
    "Fresh",
    "Not Fresh"
]

SAMPLES_PER_CLASS = 3

random.seed(42)


# ==========================================
# 2. MEMBUAT VISUALISASI
# ==========================================

fig, axes = plt.subplots(
    len(CLASSES),
    SAMPLES_PER_CLASS,
    figsize=(12, 10)
)

fig.suptitle(
    "Fish Freshness Eye Dataset",
    fontsize=18,
    fontweight="bold"
)


# ==========================================
# 3. MENAMPILKAN GAMBAR
# ==========================================

for row, class_name in enumerate(CLASSES):

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"Gambar pada kelas {class_name} tidak cukup."
        )

    selected_images = random.sample(
        images,
        SAMPLES_PER_CLASS
    )

    for col, image_name in enumerate(selected_images):

        image_path = os.path.join(
            folder_path,
            image_name
        )

        image = Image.open(image_path).convert("RGB")

        ax = axes[row, col]

        ax.imshow(image)
        ax.set_title(class_name, fontsize=11)
        ax.axis("off")


# ==========================================
# 4. MENAMPILKAN HASIL
# ==========================================

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
