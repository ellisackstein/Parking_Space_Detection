import cv2
import matplotlib as plt
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
from emptySpots import *

################ YOLO WORLD ################
# SOURCE_IMAGE_PATH = "test_img/Screenshot 2024-06-03 122621.png"
# model = YOLOWorld(model_id="yolo_world/l")
# classes = ["car", "tree", "person"]
# model.set_classes(classes)
# image = cv2.imread(SOURCE_IMAGE_PATH)
# results = model.infer(image)
# detections = sv.Detections.from_inference(results)
#
# BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
# LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
# annotated_image = image.copy()
# annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
# annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
# sv.plot_image(annotated_image, (10, 10))

################ Shira's Code ################
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
