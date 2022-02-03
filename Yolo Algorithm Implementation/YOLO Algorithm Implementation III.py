import cv2


def main():
    with open('coco.names') as file:
        class_name = [line.strip() for line in file]

    net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(IMAGE_SIZE, IMAGE_SIZE), scale=1/255, swapRB=True)
    while True:
        ret, img = video_capture.read()
        original_height, original_width, _ = img.shape

        classes, scores, boxes = model.detect(img, THRESHOLD, SUPPRESSION_THRESHOLD)

        for (class_id, confidence, box) in zip(classes, scores, boxes):
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 1)
            print(class_id)
            class_with_confidence = "{predict_class} {score}%".format(predict_class=class_name[class_id],
                                                                      score=int(confidence * 100))
            cv2.putText(img, class_with_confidence, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 2)

        cv2.imshow("Real Time Object Detection", img)
        key = cv2.waitKey(30) & 0xff

        if key == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    IMAGE_SIZE = 416
    THRESHOLD = 0.3
    SUPPRESSION_THRESHOLD = 0.3
    video_capture = cv2.VideoCapture(0)

    video_capture.set(3, 512)
    video_capture.set(4, 512)
    main()
