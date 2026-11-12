
import os
import shutil
from sklearn.model_selection import train_test_split

# ==========================================
# 1. KONFIGURASI DATASET
# ==========================================

SOURCE_PATH = "dataset/DatasetUAS_ML"
OUTPUT_PATH = "dataset/processed"

CLASSES = ["Highly Fresh", "Fresh", "Not Fresh"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ==========================================
# 2. MEMERIKSA LOKASI DATASET
# ==========================================

if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(
        f"Folder dataset tidak ditemukan: {SOURCE_PATH}"
    )

print("\n===== MEMBACA DATASET =====")


# ==========================================
# 3. MENGUMPULKAN GAMBAR BERDASARKAN KELAS
# ==========================================

images_by_class = {
    class_name: [] for class_name in CLASSES
}

for folder in os.listdir(SOURCE_PATH):

    folder_path = os.path.join(SOURCE_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    # Memeriksa nama kelas dari folder
    class_name = next(
        (
            c for c in sorted(CLASSES, key=len, reverse=True)
            if folder.endswith(c)
        ),
        None
    )

    if class_name is None:
        continue

    # Mengambil semua gambar
    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    # Menyimpan lokasi gambar dan nama file
    for image in images:

        source_file = os.path.join(folder_path, image)

        new_filename = folder + " - " + image

        images_by_class[class_name].append(
            (source_file, new_filename)
        )


# ==========================================
# 4. MENAMPILKAN JUMLAH DATASET ASLI
# ==========================================

print("\n===== PEMERIKSAAN DATASET =====")

total_original = 0

for class_name, images in images_by_class.items():

    count = len(images)

    print(f"{class_name}: {count} gambar")

    total_original += count

print("--------------------------------")
print(f"Total gambar: {total_original}")
print("================================")


# ==========================================
# 5. VALIDASI DATASET
# ==========================================

for class_name, images in images_by_class.items():

    if len(images) < 2:
        raise ValueError(
            f"Kelas {class_name} memiliki kurang dari 2 gambar."
        )

if total_original == 0:
    raise ValueError("Tidak ada gambar yang berhasil dibaca.")


# ==========================================
# 6. MEMBERSIHKAN HASIL PEMBAGIAN LAMA
# ==========================================

if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

# Membuat folder train dan test
for split in ["train", "test"]:

    for class_name in CLASSES:

        os.makedirs(
            os.path.join(OUTPUT_PATH, split, class_name),
            exist_ok=True
        )


# ==========================================
# 7. MEMBAGI DATA 80% TRAINING DAN 20% TESTING
# ==========================================

print("\n===== MEMBAGI DATASET =====")

for class_name, images in images_by_class.items():

    train_images, test_images = train_test_split(
        images,
        test_size=0.20,
        random_state=42
    )

    # Menyalin gambar training
    for source_file, filename in train_images:

        destination = os.path.join(
            OUTPUT_PATH,
            "train",
            class_name,
            filename
        )

        shutil.copy2(source_file, destination)

    # Menyalin gambar testing
    for source_file, filename in test_images:

        destination = os.path.join(
            OUTPUT_PATH,
            "test",
            class_name,
            filename
        )

        shutil.copy2(source_file, destination)

    print(
        f"{class_name}: "
        f"{len(train_images)} training, "
        f"{len(test_images)} testing"
    )


# ==========================================
# 8. MEMERIKSA HASIL PEMBAGIAN
# ==========================================

print("\n===== HASIL AKHIR DATASET =====")

total_train = 0
total_test = 0

for split in ["train", "test"]:

    print(f"\n{split.upper()}")

    for class_name in CLASSES:

        folder_path = os.path.join(
            OUTPUT_PATH,
            split,
            class_name
        )

        count = len([
            file for file in os.listdir(folder_path)
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ])

        print(f"{class_name}: {count} gambar")

        if split == "train":
            total_train += count
        else:
            total_test += count


# ==========================================
# 9. RINGKASAN
# ==========================================

print("\n================================")
print(f"Total Training : {total_train}")
print(f"Total Testing  : {total_test}")
print(f"Total Dataset  : {total_train + total_test}")
print("================================")

print("\nPembagian dataset berhasil!")
