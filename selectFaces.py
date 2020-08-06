import face_recognition
import os
import cv2

train_faces_dir = 'train'
test_faces_dir = 'test'
tolerance = 0.6
frame_thickness = 3
font_thickness = 2
model = 'cnn'

known_faces = list()
known_names = list()

print('training')

for name in os.listdir(train_faces_dir):
    for file in os.listdir(f'{train_faces_dir}/{name}'):
        path = f'{train_faces_dir}/{name}/{file}'
        print(path)
        image = cv2.imread(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print(known_faces, known_names)
print('testing')

for filename in sorted(os.listdir(test_faces_dir)):
    path = f'{test_faces_dir}/{filename}'
    print(path)
    image = cv2.imread(path)

    width = image.shape[1]
    height = image.shape[0]
    bigsize = max(width, height)
    if bigsize <= 500:
        reduce = 1
    else:
        reduce = bigsize // 500
    dsize = (int(width/reduce), int(height/reduce))
    image = cv2.resize(image, dsize)

    locations = face_recognition.face_locations(image, model=model)
    encoding = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encoding, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"match found: {match}")
