import cv2
from cv2 import Laplacian
import numpy as np
from xgboost import cv

original_image = cv2.imread('images/Trigger X Tiga.jpg', cv2.IMREAD_COLOR)

grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Use manual kernel
laplacian_kernel = np.array([
                        [0,  1, 0],
                        [1, -4, 1],
                        [0,  1, 0]
                    ])

result_image = cv2.filter2D(grayscale_image, -1, laplacian_kernel)

# use built in function opencv
result_image2 = cv2.Laplacian(grayscale_image, -1)

cv2.imshow("grayscale image", grayscale_image)
cv2.imshow("result image", result_image)
cv2.imshow("result image2", result_image2)
cv2.waitKey(0)
cv2.destroyAllWindows()