import cv2
import pygame
import time
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


pygame.init()

audio_files = {
    'can_opening': 'C:\\Users\\dharu\\OneDrive\\Desktop\\NUSProject\\classes\\can_opening\\1-41615-A-34.wav',
    'clapping': 'C:\\Users\\dharu\\OneDrive\\Desktop\\NUSProject\\classes\\clapping\\1-94036-A-22.wav'
}
def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

name = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
model = load_model("C:\\Users\\dharu\\OneDrive\\Desktop\\NUSProject\\face.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

csv_file = open("emotion_results.csv", "w")
csv_file.write("Audio_Label,Emotion\n")

for label, file_path in audio_files.items():
    print(f"Playing {label}...")
    play_audio(file_path)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (224, 224))
            face_roi = image.img_to_array(face_roi)
            face_roi = np.expand_dims(face_roi, axis=0)
            face_roi /= 255.0
            emotion_probabilities = model.predict(face_roi)
            emotion_label = np.argmax(emotion_probabilities)
            csv_file.write(f"{label},{name[emotion_label]}\n")

            emotion = f"Emotion: {name[emotion_label]}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"{label} playback and emotion recognition complete.\n")
    time.sleep(2)
csv_file.close()


pygame.quit()