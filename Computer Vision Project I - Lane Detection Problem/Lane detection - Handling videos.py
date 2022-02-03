import cv2
import numpy

video = cv2.VideoCapture('videos/lane_detection_video.mp4')

while video.isOpened():
    is_grabbed, frame = video.read()
    if not is_grabbed:
        break

    cv2.imshow("Lane Detection Video", frame)
    cv2.waitKey(20)

video.release()
cv2.destroyAllWindows()