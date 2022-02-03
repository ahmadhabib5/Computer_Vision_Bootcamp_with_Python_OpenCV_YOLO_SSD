from turtle import pos
from skimage import data, feature, transform
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from itertools import chain
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np

human_faces = fetch_lfw_people()

positive_images = human_faces.images[:10000]

rdn_idx = np.random.randint(0, positive_images.shape[0])

plt.imshow(positive_images[rdn_idx])
plt.show()

