import cv2

image = cv2.imread('images/Logo Fasilkom.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Logo Fasilkom", image)
cv2.waitKey(0)
cv2.destroyAllWindows()