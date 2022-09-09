from tensorflow.keras import models
import cv2
import face_recognition
import os
import numpy as np

dir=os.getcwd()
model=models.load_model(dir+"/model.h5")
cap=cv2.VideoCapture(0)
while True:
    _,frame=cap.read()
    cv2.imshow("Frame",frame)
    face=frame
    face=face[:, :, ::-1]
    face_locations=face_recognition.face_locations(face)
    fr=[]
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        frame=frame[top:bottom,left:right]
        frame = cv2.resize(frame, (50, 50)).flatten()
        fr.append(frame)
        fr = np.array(fr, dtype="float")/255.0
        print(fr.shape)
        prediction=model.predict(fr)
        print(prediction)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()