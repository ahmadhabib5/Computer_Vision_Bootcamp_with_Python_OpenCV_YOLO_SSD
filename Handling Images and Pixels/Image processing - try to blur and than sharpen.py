from random import gauss
import cv2
import numpy as np

original_image = cv2.imread('images/Trigger X Tiga.jpg', cv2.IMREAD_COLOR)

#blur 
blur_kernel = np.ones(shape=(9,9))/81
blur_image = cv2.filter2D(original_image, -1, blur_kernel)

# sharpen
sharpen_kernel = np.array([
                    [0, -1,  0],
                    [-1, 5, -1],
                    [0, -1,  0]
                ])
sharpen_image = cv2.filter2D(blur_kernel, -1, sharpen_kernel)

cv2.imshow("Blur Image", blur_image)
cv2.imshow("Sharpen Image", sharpen_image)
cv2.waitKey(0)
cv2.destroyAllWindows()