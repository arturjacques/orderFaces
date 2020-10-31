import face_recognition
import os
import cv2
import dlib


def reduce_image(image, biggerSize=500):
    width = image.shape[1]
    height = image.shape[0]
    bigsize = max(width, height)
    if bigsize <= biggerSize:
        reduce = 1
    else:
        reduce = bigsize // biggerSize
    dsize = (int(width / reduce), int(height / reduce))
    return cv2.resize(image, dsize)


def cut_head(image, faces, scale=1):
    cut_faces = list()
    positions = list()
    width_image = image.shape[1]
    height_image = image.shape[0]

    for face in faces:
        face_width = face.right() - face.left()
        face_height = face.bottom() - face.top()

        left = int(face.left() - face_width * (scale - 1))
        if left < 0:
            left = 0

        right = int(face.right() + face_width * (scale - 1))
        if right > width_image:
            right = width_image

        top = int(face.top() - face_height * (scale - 1))
        if top < 0:
            top = 0

        bottom = int(face.bottom() + face_height * (scale - 1))
        if bottom > height_image:
            bottom = height_image

        cut_faces.append(image[top:bottom, left:right])
        positions.append({'top': top, 'bottom': bottom, 'left': left, 'right': right})

    return cut_faces, positions


def find_match(face):
    face = reduce_image(face, biggerSize=200)
    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
    locations = face_recognition.face_locations(face, model=model)
    encoding = face_recognition.face_encodings(face, locations)
    if len(encoding)>0:
        results = face_recognition.compare_faces(known_faces, encoding[0], tolerance)
        if True in results:
            return known_names[results.index(True)]
        else:
            return 'unknown_face'
    else:
        return 'unknown_face'


train_faces_dir = 'train'
test_faces_dir = 'test'
tolerance = 0.5
model = 'cnn'

known_faces = list()
known_names = list()

for name in os.listdir(train_faces_dir):
    for file in os.listdir(f'{train_faces_dir}/{name}'):
        path = f'{train_faces_dir}/{name}/{file}'
        print(path)
        image = cv2.imread(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

detector = dlib.get_frontal_face_detector()

for filename in sorted(os.listdir(test_faces_dir)):
    path = f'{test_faces_dir}/{filename}'
    print(path)
    image = cv2.imread(path)
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    cut_faces, positions = cut_head(image, faces, scale=1)

    count = 0
    for face in cut_faces:
        cv2.imwrite(f'cuts/{filename}_{count}.jpg', face)
        count += 1

    matchs = map(find_match, cut_faces)

    print(list(matchs))
    if True in matchs:
        print('match')

