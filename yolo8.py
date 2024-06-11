from ultralytics import YOLO
import random
import cv2
import numpy as np

filename = "/Users/shiraadler/PycharmProjects/new/Parking_Space_Detection/test_img/amen.jpg"

# Load YOLO model
model = YOLO("yolov8m-seg.pt")
img = cv2.imread(filename)

# Define car classes (assuming 'car' class id is known, e.g., 2)
# You might need to check the specific model's class names to get the correct index for 'car'.
car_class_name = 'car'
car_class_id = next(key for key, value in model.names.items() if value == car_class_name)

# Set confidence threshold
conf = 0.5

# Run the model on the image
results = model.predict(img, conf=conf)

# Define color for car class (random color)
car_color = random.choices(range(256), k=3)

# Process results
for result in results:
    for mask, box in zip(result.masks.xy, result.boxes):
        if int(box.cls[0]) == car_class_id:
            points = np.int32([mask])
            cv2.fillPoly(img, points, car_color)

# Show and save the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.imwrite(filename, img)
