import cv2

SSD_INPUT_SIZE = 320
THRESHOLD = 0.5
SUPPRESSION_THRESHOLD = 0.5


def create_class_name():
    with open('class_names', 'rt') as file:
        classes_name = [line.strip() for line in file]
        return classes_name


def show_detected_objects(img, boxes_to_keep, all_bounding_boxes, object_names, class_ids):
    for index in boxes_to_keep:
        box = all_bounding_boxes[index]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(img, object_names[class_ids[index - 1]].upper(), (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)


def main():
    capture = cv2.VideoCapture('../videos/objects.mp4')
    classes = create_class_name()
    model = cv2.dnn_DetectionModel('ssd_weights.pb', 'ssd_mobilenet_coco_cfg.pbtxt')
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model.setInputParams(scale=1/127.5, size=(SSD_INPUT_SIZE, SSD_INPUT_SIZE), mean=(127.5, 127.5, 127.5), swapRB=True)

    while True:
        is_grabbed, frame = capture.read()

        if not is_grabbed:
            break

        class_ids, confidences, bbox = model.detect(frame)

        box_to_keep = cv2.dnn.NMSBoxes(bbox, confidences, THRESHOLD, SUPPRESSION_THRESHOLD)
        show_detected_objects(frame, box_to_keep, bbox, classes, class_ids)

        cv2.imshow("SSD Algorithm", frame)
        cv2.waitKey(1)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("selesai")
