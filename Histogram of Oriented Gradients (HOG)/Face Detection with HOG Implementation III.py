import cmath
from operator import pos
import random
from cv2 import resize
from skimage.io import imread
from matplotlib import image
from skimage import data, feature, transform
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from itertools import chain
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from sklearn.model_selection import GridSearchCV
import numpy as np

human_faces = fetch_lfw_people()

positive_images = human_faces.images[:10000] # positive sample

non_face_topics = ['moon', 'text', 'coins']

negative_samples = [(getattr(data, name)()) for name in non_face_topics] 

def generate_random_samples(image, num_of_generated_image=100, patch_size=positive_images[0].shape):
    extractor = PatchExtractor(patch_size=patch_size, max_patches=num_of_generated_image, random_state=42)
    patches = extractor.transform((image[np.newaxis]))
    return patches

negative_images =  np.vstack([generate_random_samples(im, 1000) for im in negative_samples])

X_train = np.array([feature.hog(image) for image in chain(positive_images, negative_images)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_images.shape[0]] = 1

parameter = {
    'C': [1,10,100,1000],
    'degree': range(1,5)
}

svc = SVC()
svc.fit(X_train, y_train)

test_image = imread(fname='images/mohammad-salah.jpg')
test_image = transform.resize(test_image, positive_images[0].shape)

test_image_hog = np.array([feature.hog(test_image)])
prediction = svc.predict(test_image_hog)
print("Prediction made by SVM: %f" % prediction)