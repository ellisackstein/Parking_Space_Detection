from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load a pretrained YOLOv8 segmentation model
model = YOLO('yolov9e-seg.pt')

# Open the image file
image_path = "test_img/Screenshot 2024-06-03 122621.png"  # Update with your image path
image = cv2.imread(image_path)

if image is not None:
    # Convert the image from BGR to RGB (matplotlib expects RGB images)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run YOLOv8 inference on the image
    results = model(image_rgb)

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
            car_boxes.append(box)

    # Plot filtered results on the image
    annotated_image = image_rgb.copy()
    for mask in car_masks:
        points = np.array(mask, dtype=np.int32)
        cv2.polylines(annotated_image, [points], isClosed=True,
                      color=(255, 0, 0), thickness=2)

    # Display the annotated image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.title("YOLOv9 Car Segmentation with Confidence > 0.5")
    plt.show()

else:
    print("Error: Unable to read the image file.")

def predict(path):
    return