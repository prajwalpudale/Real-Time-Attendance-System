import cv2
import numpy as np
import os
from datetime import datetime
import pandas as pd


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

with open("labels.txt", "r") as f:
    labels = {int(line.split(":")[0]): line.strip().split(":")[1] for line in f}

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

attendance = set()
csv_file = "attendance.csv"
while True:
    ret, img = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(face)
        name = labels.get(id_, "Unknown")

        if confidence < 70:
            if name not in attendance:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df = pd.DataFrame([[name, now]], columns=["Name", "Time"])
                if os.path.exists(csv_file):
                    df.to_csv(csv_file, mode='a', header=False, index=False)
                else:
                    df.to_csv(csv_file, index=False)
                attendance.add(name)

            label = f"{name} ({int(confidence)}%)"
        else:
            label = "Unknown"

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Attendance System", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print(" Attendance marked. Press 'q' to close.")
