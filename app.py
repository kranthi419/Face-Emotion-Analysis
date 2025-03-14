import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the pre-trained model and Haar cascade classifier
face_classifier = cv2.CascadeClassifier(r'HaarcascadeclassifierCascadeClassifier.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


# Function to detect emotions in a frame
def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)
    if not len(faces):
        cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            for i, (emotion, score) in enumerate(zip(emotion_labels, prediction)):
                label_position = (x + w + 10, y + (i * 20))
                color = (0, 255, 0) if score == max(prediction) else (0, 0, 255)
                cv2.putText(frame, f"{emotion}: {score:.2f}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


# Streamlit UI
st.title("Real-Time Facial Emotion Detection")
run = st.toggle('Start Analysis...')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture image")
        break
    frame = detect_emotions(frame)
    FRAME_WINDOW.image(frame, channels='BGR')

cap.release()
