import numpy as np
import cv2

image = cv2.imread('images/unsharp_bird.jpg')

kernel = np.array([
            [0, -1,  0],
            [-1, 5, -1],
            [0, -1,  0]
        ])

sharpen_iamge = cv2.filter2D(image, -1, kernel)

cv2.imshow("original image", image)
cv2.imshow("sharpen image", sharpen_iamge)
cv2.waitKey(0)
cv2.destroyAllWindows()