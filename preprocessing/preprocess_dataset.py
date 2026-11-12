
import os
import shutil
from PIL import Image

# ==========================================
# 1. KONFIGURASI
# ==========================================

SOURCE_PATH = "dataset/processed"
OUTPUT_PATH = "dataset/preprocessed"

IMAGE_SIZE = (128, 128)

CLASSES = [
    "Highly Fresh",
    "Fresh",
    "Not Fresh"
]

SPLITS = ["train", "test"]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ==========================================
# 2. MEMERIKSA DATASET
# ==========================================

if not os.path.exists(SOURCE_PATH):
    raise FileNotFoundError(
        f"Dataset tidak ditemukan: {SOURCE_PATH}"
    )

print("\n===== MEMULAI PREPROCESSING =====")


# ==========================================
# 3. MEMBERSIHKAN HASIL LAMA
# ==========================================

if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)


# ==========================================
# 4. MEMBUAT FOLDER OUTPUT
# ==========================================

for split in SPLITS:
    for class_name in CLASSES:

        folder_path = os.path.join(
            OUTPUT_PATH,
            split,
            class_name
        )

        os.makedirs(folder_path, exist_ok=True)


# ==========================================
# 5. PREPROCESSING GAMBAR
# ==========================================

total_success = 0
total_failed = 0

for split in SPLITS:

    print(f"\nMemproses folder {split.upper()}...")

    for class_name in CLASSES:

        source_folder = os.path.join(
            SOURCE_PATH,
            split,
            class_name
        )

        output_folder = os.path.join(
            OUTPUT_PATH,
            split,
            class_name
        )

        images = [
            file for file in os.listdir(source_folder)
            if file.lower().endswith(IMAGE_EXTENSIONS)
        ]

        success = 0
        failed = 0

        for image_name in images:

            source_file = os.path.join(
                source_folder,
                image_name
            )

            output_file = os.path.join(
                output_folder,
                image_name
            )

            try:
                # Membuka gambar
                with Image.open(source_file) as image:

                    # Mengubah gambar menjadi RGB
                    image = image.convert("RGB")

                    # Resize gambar menjadi 128 x 128
                    image = image.resize(
                        IMAGE_SIZE,
                        Image.Resampling.LANCZOS
                    )

                    # Menyimpan hasil preprocessing
                    image.save(
                        output_file,
                        format="JPEG",
                        quality=95
                    )

                success += 1
                total_success += 1

            except Exception as error:

                print(
                    f"Gagal memproses {image_name}: {error}"
                )

                failed += 1
                total_failed += 1

        print(
            f"{class_name}: "
            f"{success} berhasil, {failed} gagal"
        )


# ==========================================
# 6. RINGKASAN HASIL
# ==========================================

print("\n===== HASIL PREPROCESSING =====")

print(f"Ukuran gambar : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
print(f"Total berhasil: {total_success}")
print(f"Total gagal   : {total_failed}")

print("================================")

if total_failed == 0:
    print("Preprocessing berhasil!")
else:
    print("Preprocessing selesai, tetapi ada gambar gagal diproses.")
