import cv2
import numpy as np

IMAGE_SIZE = 320
THRESHOLD = 0.5
SUPPRESSION_THRESHOLD = 0.4
video_capture = cv2.VideoCapture(0)

video_capture.set(3, 512)
video_capture.set(4, 512)


def find_object(model_outputs):
    bounding_box_location = []
    class_ids = []
    confidence_values = []
    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_idx = np.argmax(class_probabilities)
            confidence = class_probabilities[class_idx]
            if confidence > THRESHOLD:
                w, h = int(prediction[2] * IMAGE_SIZE), int(prediction[3] * IMAGE_SIZE)
                x, y = int(prediction[0] * IMAGE_SIZE - w / 2), int(prediction[2] * IMAGE_SIZE - h / 2)
                bounding_box_location.append([x, y, w, h])
                class_ids.append(class_idx)
                confidence_values.append(float(confidence))
    bbox_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_location, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)
    return bbox_indexes_to_keep, bounding_box_location, class_ids, confidence_values


def show_detected_images(img, bounding_box_ids, all_bounding_boxes, class_ids, confidence_values, width_ratio,
                         height_ratio, classes):
    for index in bounding_box_ids:
        bounding_box = all_bounding_boxes[index]
        x, y, w, h = [int(val) for val in bounding_box]
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)
        print(index)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
        class_with_confidence = classes[index - 1] + str(int(confidence_values[index - 1] * 100)) + "%"
        cv2.putText(img, class_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0), 1)


def main():
    with open('class_names') as file:
        classes = [line.strip() for line in file]

    model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    while True:
        ret, img = video_capture.read()
        original_height, original_width, _ = img.shape

        blob = cv2.dnn.blobFromImage(image=img, scalefactor=1 / 255, size=(IMAGE_SIZE, IMAGE_SIZE), swapRB=True,
                                     crop=False)
        model.setInput(blob)

        layer_name = model.getLayerNames()
        output_names = [layer_name[index - 1] for index in model.getUnconnectedOutLayers()]
        outputs = model.forward(output_names)
        predicted_object, bbox_locations, class_label, conf_values = find_object(outputs)
        show_detected_images(img, predicted_object, bbox_locations,
                             class_label, conf_values, original_width / IMAGE_SIZE,
                             original_height / IMAGE_SIZE, classes)
        cv2.imshow("Real Time Object Detection", img)
        key = cv2.waitKey(30) & 0xff

        if key == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
