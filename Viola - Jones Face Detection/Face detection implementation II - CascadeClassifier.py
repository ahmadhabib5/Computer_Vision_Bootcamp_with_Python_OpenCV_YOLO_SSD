import cv2
from cv2 import data
import os
import matplotlib.pyplot as plt


print(os.listdir(data.haarcascades))

haar_path = data.haarcascades

image = cv2.imread('images/aguero-debruyne.jpg')

grayscale_iamge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cascade_classifier = cv2.CascadeClassifier(os.path.join(haar_path, 'haarcascade_profileface.xml'))

detect_face = cascade_classifier.detectMultiScale(grayscale_iamge, scaleFactor=1.1,
                                                  minNeighbors=10,
                                                  minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, width, height) in detect_face:
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()            