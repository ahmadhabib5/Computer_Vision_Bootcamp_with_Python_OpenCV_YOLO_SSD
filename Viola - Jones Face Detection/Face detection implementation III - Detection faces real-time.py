import cv2
import os
from cv2 import data
import matplotlib.pyplot as plt
from xgboost import cv

haar_path = data.haarcascades

cascade_classifier_profile = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_profileface.xml'))
cascade_classifier_frontal = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_frontalface_alt2.xml'))
video_capture = cv2.VideoCapture(0)

video_capture.set(3, 1000)  # WIDTH
video_capture.set(4, 1000)  # HEIGHT

while True:
    ret, img = video_capture.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detect_face = cascade_classifier_profile.detectMultiScale(gray_img, scaleFactor=1.3,
                                                              minNeighbors=10,
                                                              minSize=(30, 30))

    if len(detect_face) == 0:
        detect_face = cascade_classifier_frontal.detectMultiScale(gray_img, scaleFactor=1.3,
                                                                  minNeighbors=10,
                                                                  minSize=(30, 30))
    for (x, y, width, height) in detect_face:
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 5)

    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(30) & 0xff

    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
