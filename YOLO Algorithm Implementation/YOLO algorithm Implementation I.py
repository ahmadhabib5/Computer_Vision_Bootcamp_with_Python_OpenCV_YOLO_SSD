import cv2
import numpy as np

IMAGE_SIZE = 320
THRESHOLD = 0.5
SUPPRESSION_THRESHOLD = 0.5


def show_detected_images(img, bounding_box_idx, all_bounding_boxes, class_idx,
                         confidence_values, width_ratio, height_ratio):
    for index in bounding_box_idx:
        bounding_box = all_bounding_boxes[index]
        x, y, w, h = int(bounding_box[0]), int(bounding_box[1]), int(bounding_box[2]), int(bounding_box[3])
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)

        if class_idx[index] == 2:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'CAR ' + str(int(confidence_values[index] * 100)) + "%"
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 0), 1)
        if class_idx[index] == 0:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'PERSON ' + str(int(confidence_values[index] * 100)) + "%"
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)
        if class_idx[index] == 3:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'MOTORCYCLE ' + str(int(confidence_values[index] * 100)) + "%"
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (50, 205, 50), 1)
        if class_idx[index] == 9:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            class_with_confidence = 'TRAFFIC LIGHT ' + str(int(confidence_values[index] * 100)) + "%"
            cv2.putText(img, class_with_confidence, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 255, 0), 1)


def find_object(model_outputs):
    bounding_box_location = []
    class_idx = []
    confidence_values = []

    for output in model_outputs:
        for prediction in output:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]
            # x, y, w, h, confidence, class_probabilities...
            if confidence > THRESHOLD:
                width, height = int(prediction[2]*IMAGE_SIZE), int(prediction[3]*IMAGE_SIZE)
                x, y = int((prediction[0]*IMAGE_SIZE)-(width/2)), int((prediction[1]*IMAGE_SIZE)-(height/2))
                bounding_box_location.append([x, y, width, height])
                class_idx.append(class_id)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_location, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)
    return box_indexes_to_keep, bounding_box_location, class_idx, confidence_values


def main():
    image = cv2.imread('../images/traffic.jpg')
    original_height, original_width, _ = image.shape

    # 0 person, 2 car, 3 motorcycle, 9 traffic light
    classes = ["person", "car", "motorcycle", "traffic light"]

    nn = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    nn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    nn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    blob = cv2.dnn.blobFromImage(image, 1 / 255, (IMAGE_SIZE, IMAGE_SIZE), True, crop=False)
    nn.setInput(blob)

    layer_name = nn.getLayerNames()
    output_names = [layer_name[index-1] for index in nn.getUnconnectedOutLayers()]
    outputs = nn.forward(output_names)
    predicted_object, bbox_locations, class_label, conf_values = find_object(outputs)
    show_detected_images(image, predicted_object, bbox_locations,
                         class_label, conf_values, original_width/IMAGE_SIZE,
                         original_height/IMAGE_SIZE)

    cv2.imshow("YOLO", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
