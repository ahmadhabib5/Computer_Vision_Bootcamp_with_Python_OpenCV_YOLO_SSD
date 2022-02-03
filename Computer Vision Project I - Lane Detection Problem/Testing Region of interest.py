import cv2
import numpy as np

def region_of_interest(image, region_points):
    mask = np.zeros_like(image)

    cv2.fillPoly(mask, region_points, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


image = cv2.imread('images/unsharp_bird.jpg')
height, width, _ = image.shape

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


region_of_interest_verties = [
    [0, height*0.7],
    [0, height],
    [width, height],
    [width, height*0.7],
    [width/2, height*0.5]
]
cropped_image = region_of_interest(grayscale_image, np.array([region_of_interest_verties], np.int32))

cv2.imshow("test", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()