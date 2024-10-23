import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# Path to images for attendance
path = 'ImagesAttendence'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

# Load the images and their corresponding class names
for i in myList:
    currentImage = cv2.imread(f'{path}/{i}')
    images.append(currentImage)
    classNames.append(os.path.splitext(i)[0])
print(classNames)


# Function to find and encode faces from the images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except IndexError:
            print("Face not found in image, skipping:", img)
            continue
    return encodeList
# Function to Mark Attendence
def markAttendece(name):
    with open('Attendence.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dateSrting = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateSrting}')



# Get known face encodings
encodeListKnown = findEncodings(images)
print('Encoding completed!')

# Start the webcam and recognize faces in real-time
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    if not success:
        print("Failed to grab frame")
        break
    img = cv2.flip(img, 1)

    # Resize frame for faster face recognition processing
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find all face locations and face encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Loop through each face found in the frame
    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        # Find the best match
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()


            # Scale the face location back up since we resized the frame
            top, right, bottom, left = faceLoc
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            # Draw a rectangle around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw the name of the person
            cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            markAttendece(name)


    cv2.imshow('Webcam', img)

    # Exit on pressing 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()