import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []
ids = []
names = {}
id_count = 0

dataset_path = "dataset"

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    names[id_count] = person_name
    print(f"Loading images for '{person_name}'...")

    for img_name in os.listdir(person_folder):
        path = os.path.join(person_folder, img_name)

        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {img_name}")
            continue

        gray_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if gray_img is None:
            print(f"Skipping unreadable image: {path}")
            continue

        faces.append(gray_img)
        ids.append(id_count)

    id_count += 1

if len(faces) == 0 or len(ids) == 0:
    print("No valid training data found. Please check your dataset.")
    exit()

print("Training model...")
recognizer.train(faces, np.array(ids))

recognizer.save("trainer.yml")
print("Model saved as 'trainer.yml'")

with open("labels.txt", "w") as f:
    for id, name in names.items():
        f.write(f"{id}:{name}\n")

print("Label map saved as 'labels.txt'")
print("Model trained and saved successfully.")
