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

