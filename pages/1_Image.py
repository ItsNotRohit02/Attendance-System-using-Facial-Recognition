import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd
from PIL import Image


st.set_page_config(page_title="Face Recognition using Image", page_icon="📷")

def load_face_encodings():
    df = pd.read_csv('EncodedFaces.csv')
    face_encodings = np.array(df.iloc[:, 0].apply(eval).tolist())
    names = df.iloc[:, 1].tolist()
    return face_encodings, names


def recognize_faces(uploaded_image, face_encodings, names):
    image = np.array(uploaded_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image)
    face_encodings_test = face_recognition.face_encodings(image, face_locations)
    face_labels = []

    for face_encoding in face_encodings_test:
        matches = face_recognition.compare_faces(face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            accuracy = 1 - face_distances[best_match_index]
            if accuracy > 0.5:
                name = names[best_match_index]
        face_labels.append(name)
    return face_locations, face_labels


def draw_boxes(image, face_locations, face_labels):
    for (top, right, bottom, left), name in zip(face_locations, face_labels):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(image, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)
    return image


def main():
    st.title("Face Recognition from an Image")
    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        face_encodings, names = load_face_encodings()
        image = Image.open(uploaded_image)
        face_locations, face_labels = recognize_faces(image, face_encodings, names)
        image = np.array(image)
        image_with_boxes = draw_boxes(image, face_locations, face_labels)
        st.image(image_with_boxes, channels="RGB", caption="Recognized Faces")
        st.write("Attendance Marked for")
        for name in face_labels:
            if name != 'Unknown':
                st.write(name)
    st.caption("Made by Code Knights")


if __name__ == '__main__':
    main()
