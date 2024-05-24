###### Image Processing #######

# import cv2
# import numpy as np
#
# def detect_motion(frame1, frame2, threshold=50):
#     gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#
#     diff = cv2.absdiff(gray1, gray2)
#     _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
#
#     # Find contours of the thresholded image to detect areas of motion
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     motion_level = np.sum(thresh) / 255
#     return motion_level, contours
#
# def classify_motion(motion_level, moving_threshold=5000):
#     if motion_level > moving_threshold:
#         return "passing"
#     else:
#         return "parked"
#
# def draw_motion_contours(frame, contours):
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:  # Filter out small contours
#             x, y, w, h = cv2.boundingRect(contour)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
# # Read the frames
# frame1 = cv2.imread("20240426_125431.jpg")
# frame2 = cv2.imread("20240426_125432.jpg")
#
# # Detect motion
# motion_level, contours = detect_motion(frame1, frame2)
#
# # Classify motion and draw contours if motion is detected
# motion_type = classify_motion(motion_level)
# if motion_type == "passing":
#     draw_motion_contours(frame2, contours)
#
# # Display the result
# cv2.imshow("Motion Detection", frame2)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()
#
# # Optionally save the result to a file
# cv2.imwrite("motion_detected.jpg", frame2)

import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld


def extract_detections(image_path, model):
    image = cv2.imread(image_path)
    results = model.infer(image)
    detections = sv.Detections.from_inference(results)
    return detections, image


def compare_detections(detections1, detections2):
    # Convert detections to sets of tuples for easy comparison
    boxes1 = {tuple(det) for det in detections1}  # Convert numpy arrays to tuples
    boxes2 = {tuple(det) for det in detections2}  # Convert numpy arrays to tuples

    # Find unique detections in each set
    unique_to_image1 = boxes1 - boxes2
    unique_to_image2 = boxes2 - boxes1

    return unique_to_image1, unique_to_image2

def annotate_image(image, unique_boxes):
    for box in unique_boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for unique boxes


def moving_vs_stat(path_1, path_2):
    model = YOLOWorld(model_id="yolo_world/l")
    classes = ["car", "tree", "person"]
    model.set_classes(classes)

    detections1, image1 = extract_detections(path_1, model)
    detections2, image2 = extract_detections(path_2, model)

    unique_to_image1, unique_to_image2 = compare_detections(detections1, detections2)

    annotate_image(image1, unique_to_image1)
    annotate_image(image2, unique_to_image2)

    sv.plot_image(image1, (10, 10))  # Display first image with unique boxes
    sv.plot_image(image2, (10, 10))  # Display second image with unique boxes


# Example usage
moving_vs_stat("20240426_125431.jpg", "20240426_125432.jpg")