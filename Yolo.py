from ultralytics import YOLO
import cv2
from ExtractEmptySpots import present_results


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
    present_results(car_boxes,path)
    return car_boxes, car_masks, image

#predict("static/res/image_latest.jpg")