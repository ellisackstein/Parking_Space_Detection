import cv2
import matplotlib as plt
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
from emptySpots import *
from Preprocessing import preprocessing

PARKING_AREA = {1: [], 2: [], 3: []}

SOURCE_IMAGE_PATH = "test_img/Screenshot 2024-06-03 122621.png"
model = YOLOWorld(model_id="yolo_world/l")
classes = ["car", "tree", "person"]
model.set_classes(classes)
image = cv2.imread(SOURCE_IMAGE_PATH)
results = model.infer(image)
detections = sv.Detections.from_inference(results)

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
annotated_image = image.copy()
annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
sv.plot_image(annotated_image, (10, 10))


# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt
#
# # Load a pretrained YOLOv8 segmentation model
# model = YOLO('yolov9e-seg.pt')
#
# # Open the image file
# image_path = "test_img/Screenshot 2024-06-03 122621.png"  # Update with your image path
# image = cv2.imread(image_path)
#
# if image is not None:
#     # Convert the image from BGR to RGB (matplotlib expects RGB images)
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#     # Run YOLOv8 inference on the image
#     results = model(image_rgb)
#
#     # Visualize the results on the image
#     annotated_image = results[0].plot(show=False)  # Use show=False to get the image with annotations
#
#     # Display the annotated image
#     plt.figure(figsize=(12, 8))
#     plt.imshow(annotated_image)
#     plt.axis('off')
#     plt.title("YOLOv9 Inference")
#     plt.show()
# else:
#     print("Error: Unable to read the image file.")

# Step 1: Preprocessing video of destination
# path = preprocessing("Scenes/scene1")

# Step 2: Cancelling moving cars


# Step 3: Distinguishing the parking areas


# Step 4: Detecting empty parking spots


# # Determine the minimum car width
# min_car_width = find_smallest_car(detections)
#
# # Extract car bounding boxes
# car_detections = extract_car_detections(detections)
#
# # Use the minimum car width as the required width for a parking spot
# free_spots = free_parking_between_cars(car_detections, min_car_width)
#
# # Create a copy of the image to draw annotations
# annotated_image = image.copy()
#
# # Annotate the image with bounding boxes and labels for free parking spots
# for spot in free_spots:
#     left_car, right_car = spot
#     x1 = int(left_car[2])
#     y1 = int(left_car[1])
#     x2 = int(right_car[0])
#     y2 = int(right_car[3])
#     cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(annotated_image, 'Free Spot', (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#
# # Display the result
# sv.plot_image(annotated_image, (10, 10))
