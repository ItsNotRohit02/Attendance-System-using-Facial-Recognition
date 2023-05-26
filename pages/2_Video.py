import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd


def load_face_encodings():
    df = pd.read_csv('EncodedFaces.csv')
    face_encodings = np.array(df.iloc[:, 0].apply(eval).tolist())
    names = df.iloc[:, 1].tolist()
    return face_encodings, names


def recognize_faces_in_video(video_path, face_encodings, names):
    cap = cv2.VideoCapture(video_path)
    face_locations = []
    face_labels = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        current_face_locations = face_recognition.face_locations(rgb_frame)
        current_face_encodings = face_recognition.face_encodings(rgb_frame, current_face_locations)
        current_face_labels = ["Unknown"] * len(current_face_locations)
        for i, face_encoding in enumerate(current_face_encodings):
            matches = face_recognition.compare_faces(face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                accuracy = 1 - face_distances[best_match_index]
                if accuracy > 0.5:
                    current_face_labels[i] = names[best_match_index]

        face_locations.append(current_face_locations)
        face_labels.append(current_face_labels)

        for (top, right, bottom, left), name in zip(current_face_locations, current_face_labels):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (0, 0, 0), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return face_locations, face_labels


def main():
    st.title("Face Recognition from a Video")
    uploaded_video = st.file_uploader("Upload a video", type=['mp4'])

    if uploaded_video is not None:
        face_encodings, names = load_face_encodings()
        with open("temp.mp4", "wb") as f:
            f.write(uploaded_video.read())
        face_locations, face_labels = recognize_faces_in_video("temp.mp4", face_encodings, names)
        st.write("Attendance Marked for")
        recognized_names = set()
        for frame_labels in face_labels:
            for name in frame_labels:
                if name != "Unknown" and name not in recognized_names:
                    recognized_names.add(name)
                    st.write(name)


if __name__ == '__main__':
    main()
