import cv2
import numpy as np

original_image = cv2.imread('images/Trigger X Tiga.jpg', cv2.IMREAD_COLOR)

kernel = np.ones(shape=(5,5))/25 # 25 is total pixel of kernel, simply is width x height of kernel
kernel2 = np.ones(shape=(9,9))/81 

blur_image = cv2.filter2D(original_image, -1, kernel)
blur_image2 = cv2.filter2D(original_image, -1, kernel2)

cv2.imshow("original image", original_image)
cv2.imshow("blurred image", blur_image)
cv2.imshow("blurred image 2", blur_image2)

cv2.waitKey(0)
cv2.destroyAllWindows()

