import cv2
import face_recognition
import numpy as np
import csv
import datetime

known_face_encodings = []
known_face_names = []

with open('EncodedFaces.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        face_encoding = np.array([float(x) for x in row[0].split(',')])
        name = row[1]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

face_locations = []
face_encodings = []
face_names = []
video_capture = cv2.VideoCapture(0)

attendance_file = 'Attendance.csv'

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []

    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = known_face_names[best_match_index]
        accuracy = (1 - face_distances[best_match_index]) * 100

        if accuracy >= 50:
            face_names.append(f"{name} ({accuracy:.2f}%)")

            with open(attendance_file, 'r') as file:
                reader = csv.reader(file)
                name_exists = any(name in row for row in reader)

            with open(attendance_file, 'a', newline='') as file:
                if not name_exists:
                    writer = csv.writer(file)
                    writer.writerow([name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        else:
            face_names.append("Unknown")

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left, bottom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
