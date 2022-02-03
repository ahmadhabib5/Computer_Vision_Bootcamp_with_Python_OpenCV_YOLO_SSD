import cv2
from cv2 import data
import os
import matplotlib.pyplot as plt

haar_feature_path = data.haarcascades

# print(os.listdir(haar_feature_path))

cascade_classifier = cv2.CascadeClassifier(os.path.join(haar_feature_path, 'haarcascade_frontalface_alt.xml'))

image = cv2.imread('images/mohammad-salah.jpg')

grayscale_iamge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detect_face = cascade_classifier.detectMultiScale(grayscale_iamge, scaleFactor=1.1,
                                                  minNeighbors=10,
                                                  minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, width, height) in detect_face:
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 10)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()                                                