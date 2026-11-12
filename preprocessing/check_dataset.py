
import os

# Lokasi dataset
DATASET_PATH = "dataset/DatasetUAS_ML"

# Kategori kesegaran
classes = {
    "Highly Fresh": 0,
    "Fresh": 0,
    "Not Fresh": 0
}

# Menghitung gambar berdasarkan kategori
for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder)

    if os.path.isdir(folder_path):

        for class_name in classes:

            if folder.endswith(class_name):

                images = [
                    file for file in os.listdir(folder_path)
                    if file.lower().endswith((".jpg", ".jpeg", ".png"))
                ]

                classes[class_name] += len(images)

# Menampilkan hasil
print("===== DATASET FISH FRESHNESS =====")

total = 0

for class_name, count in classes.items():
    print(f"{class_name}: {count} gambar")
    total += count

print("---------------------------------")
print(f"Total gambar: {total}")
print("=================================")
