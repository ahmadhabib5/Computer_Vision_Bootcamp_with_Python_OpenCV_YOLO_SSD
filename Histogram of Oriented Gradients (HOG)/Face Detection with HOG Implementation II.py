import cmath
import random
from skimage import data, feature, transform
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from itertools import chain
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
import numpy as np

human_faces = fetch_lfw_people()

positive_images = human_faces.images[:10000] # positive sample

non_face_topics = ['moon', 'text', 'coins', 'horse']

negative_images = [(getattr(data, name)()) for name in non_face_topics] 

def generate_random_samples(image, num_of_generate_image=100, patch_size=positive_images[0].shape):
    extractor = PatchExtractor(patch_size=patch_size, max_patches=num_of_generate_image, random_state=42)
    patches = extractor.transform((image[np.newaxis]))
    return patches

negative_images =  np.vstack([generate_random_samples(im, 2000) for im in negative_images])



fig, ax = plt.subplots(5, 5)

for i, axis in enumerate(ax.flat):
    rdn_idx = np.random.randint(0, negative_images.shape[0])
    axis.imshow(negative_images[rdn_idx], cmap='gray')
    axis.axis('off')

plt.show()