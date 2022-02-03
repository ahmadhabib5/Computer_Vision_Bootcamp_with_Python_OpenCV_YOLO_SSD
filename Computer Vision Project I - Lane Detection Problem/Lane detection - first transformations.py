import cv2
from cv2 import cvtColor
import numpy as np


def get_detect_lanes(image, filter='laplacian'):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if filter.lower() == 'laplacian':
        edge_kernel = np.array([
                        [0, 1, 0],
                        [1, -4, 1],
                        [0 ,1, 0]
                    ])
        result_image = cv2.filter2D(grayscale_image, -1, edge_kernel)
    elif filter.lower() == 'canny':
        result_image = cv2.Canny(grayscale_image, 70, 100)
    return result_image

video = cv2.VideoCapture('videos/lane_detection_video.mp4')

while video.isOpened():
    is_grabbed, frame = video.read()
    if not is_grabbed:
        break
    
    frame = get_detect_lanes(frame, 'canny')

    cv2.imshow("Lane Detection Video", frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()