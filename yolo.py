# yolo9
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from emptySpots import present_results

# yolo world
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld


def predict(path):
    # Load a pretrained YOLOv8 segmentation model
    model = YOLO('yolov9e-seg.pt')

    # Open the image file
    image = cv2.imread(path)
    car_boxes, car_masks = [], []

    if image is not None:
        # Convert the image from BGR to RGB (matplotlib expects RGB images)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference on the image
        results = model.predict(image_rgb)

        # Define car class id (assuming 'car' class id is known, e.g., 2)
        car_class_name = 'car'
        car_class_id = next(
            key for key, value in model.names.items() if value == car_class_name)

        # Create lists to hold car masks and boxes
        car_masks = []
        car_boxes = []

        # Filter results to include only cars with confidence > 0.5
        for mask, box in zip(results[0].masks.xy, results[0].boxes):
            if int(box.cls[0]) == car_class_id and box.conf[0] > 0.5:
                car_masks.append(mask)
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # Convert tensor to list
                car_boxes.append((x1, y1, x2, y2))

    else:
        print("Error: Unable to read the image file.")
    return car_boxes, car_masks, image


def predict_yolo_world(path):
    model = YOLOWorld(model_id="yolo_world/l")
    classes = ["car"]
    model.set_classes(classes)
    image = cv2.imread(path)
    results = model.infer(image)
    detections = sv.Detections.from_inference(results)

    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=5)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
    annotated_image = image.copy()
    annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
    # sv.plot_image(annotated_image, (10, 10))

    return detections, annotated_image