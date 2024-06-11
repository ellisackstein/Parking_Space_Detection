import cv2
import numpy as np
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

# Load and configure the model
SOURCE_IMAGE_PATH = "test_img/20240426_122224.jpg"
model = YOLOWorld(model_id="yolo_world/l")
classes = ["car", "tree", "person"]
model.set_classes(classes)

# Read the image
image = cv2.imread(SOURCE_IMAGE_PATH)

# Perform inference
results = model.infer(image)
detections = sv.Detections.from_inference(results)

# Extract the center points of the detections
centers = []
for bbox in detections.xyxy:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    centers.append((center_x, center_y))

# Convert the centers to a NumPy array
centers = np.array(centers)

# Calculate the median y-coordinate
mean_y = np.mean(centers[:, 1])

# Prepare to draw the horizontal line on the image
image_height, image_width, _ = image.shape
line_y = int(mean_y)
line_start = (0, line_y)
line_end = (image_width, line_y)

# Annotate the image with bounding boxes and labels
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
annotated_image = image.copy()
annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)

# Draw the horizontal separator line
cv2.line(annotated_image, line_start, line_end, (0, 0, 255), 8)

# Display the annotated image
sv.plot_image(annotated_image, (10, 10))
