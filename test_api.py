
import requests

url = "http://127.0.0.1:5000/predict"

image_path = "dataset/preprocessed/test/Fresh/Chanos Chanos - Fresh - IMG_20191002_063314.jpg"

with open(image_path, "rb") as image:
    response = requests.post(
        url,
        files={"file": image}
    )

print("Status Code:", response.status_code)
print("Hasil Prediksi:")
print(response.json())
