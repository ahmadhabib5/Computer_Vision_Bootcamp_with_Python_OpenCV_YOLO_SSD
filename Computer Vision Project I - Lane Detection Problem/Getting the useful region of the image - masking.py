import cv2
import numpy as np

def draw_the_lines(image, lines):
    lines_image = np.zeros(shape=(image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255,0,0), thickness=3)
    
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)
    return image_with_lines

def region_of_interest(image, region_points):
    mask = np.zeros_like(image)

    cv2.fillPoly(mask, region_points, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def get_detect_lanes(image, filter='laplacian'):

    height, width, _ = image.shape

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

    region_of_interest_verties = [
        [0, height],
        [width/2, height*0.6],
        [width, height]
    ]
    cropped_image = region_of_interest(result_image, np.array([region_of_interest_verties], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi/ 180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=150)

    image_with_lines = draw_the_lines(image, lines)

    return image_with_lines

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