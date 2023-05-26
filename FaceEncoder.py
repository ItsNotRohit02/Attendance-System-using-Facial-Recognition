import os
import csv
import face_recognition

image_folder = 'Augmented_Images'
known_face_encodings = {}

for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        name = os.path.splitext(filename)[0].split(' ')[0].split('_')[0]
        image_path = os.path.join(image_folder, filename)
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if name in known_face_encodings:
            known_face_encodings[name].extend(face_encodings)
        else:
            known_face_encodings[name] = face_encodings

with open('EncodedFaces.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for name, face_encodings in known_face_encodings.items():
        for face_encoding in face_encodings:
            face_encoding_str = ','.join(map(str, face_encoding))
            writer.writerow([face_encoding_str, name])
