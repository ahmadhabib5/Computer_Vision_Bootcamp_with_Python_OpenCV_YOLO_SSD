from skimage import data, feature
import matplotlib.pyplot as plt

image = plt.imread('images/mohammad-salah.jpg')

hog_vector, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8,8),
                                    cells_per_block=(2,2), block_norm='L2',
                                    visualize=True)

fig, ax = plt.subplots(1, 2, figsize=(12,9), subplot_kw=dict(xticks=[], yticks=[]))

ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[1].imshow(hog_image)
ax[1].set_title("HOG Image")

plt.show()