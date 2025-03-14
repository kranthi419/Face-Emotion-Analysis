import av
import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load the pre-trained model and Haar cascade classifier
face_classifier = cv2.CascadeClassifier(r'HaarcascadeclassifierCascadeClassifier.xml')
classifier = load_model(r'model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


class EmotionDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        if not len(faces):
            cv2.putText(img, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
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
                    cv2.putText(img, f"{emotion}: {score:.2f}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(img, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img


# Streamlit UI
st.title("Real-Time Facial Emotion Detection")
webrtc_streamer(key="emotion-detection", video_transformer_factory=EmotionDetector)
