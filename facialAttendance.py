import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime

video_capture = cv2.VideoCapture(0)

wang_image = face_recognition.load_image_file("photos/dylan.jpg")
wang_encoding = face_recognition.face_encoding(wang_image)[0]

chen_image = face_recognition.load_image_file("photos/xingxu.jpg")
chen_encoding = face_recognition.face_encoding(chen_image)[0]

hong_image = face_recognition.load_image_file("photos/joshua.jpg")
hong_encoding = face_recognition.face_encoding(hong_image)[0]

zhao_image = face_recognition.load_image_file("photos/lusi.jpg")
zhao_encoding = face_recognition.face_encoding(zhao_image)[0]

known_face_encoding = [
    wang_encoding,
    chen_encoding,
    hong_encoding,
    zhao_encoding
]

know_faces_name = [
    "Dylan Wang",
    "Chen Xingxu",
    "Joshua Hong",
    "Lusi Zhao"
]

students = know_faces_name.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = know_faces_name[best_match_index]
