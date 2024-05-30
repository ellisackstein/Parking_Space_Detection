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


import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import cv2
import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box1, box2: Bounding boxes in the format [x1, y1, x2, y2]

    Returns:
        iou: IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def compare_detections(detections1, detections2, iou_threshold=0.70):
    """
    Compare two sets of detections and find unique detections in each set based on IoU.

    Args:
        detections1, detections2: Detections objects with `xyxy` attribute containing bounding boxes.
        iou_threshold: IoU threshold to consider two boxes as the same.

    Returns:
        unique_to_image1: List of detections unique to image 1.
        unique_to_image2: List of detections unique to image 2.
    """
    boxes1 = detections1.xyxy
    boxes2 = detections2.xyxy

    unique_to_image1 = []
    unique_to_image2 = boxes2.copy()

    for box1 in boxes1:
        found = False
        for box2 in boxes2:
            iou = calculate_iou(box1, box2)
            if iou >= iou_threshold:
                found = True
                unique_to_image2 = np.array([box for box in unique_to_image2 if not np.array_equal(box, box2)])
                break
        if not found:
            unique_to_image1.append(box1)

    return unique_to_image1, unique_to_image2


def moving_vs_stat(path_1, path_2):
    # Load the model and set classes
    model = YOLOWorld(model_id="yolo_world/l")
    classes = ["car", "tree", "person"]
    model.set_classes(classes)

    # Process the first image
    image1 = cv2.imread(path_1)
    results1 = model.infer(image1)
    detections1 = sv.Detections.from_inference(results1)

    # Process the second image
    image2 = cv2.imread(path_2)
    results2 = model.infer(image2)
    detections2 = sv.Detections.from_inference(results2)

    # Compare detections
    unique_to_image1, unique_to_image2 = compare_detections(detections1, detections2)

    # Annotate and display images
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

    annotated_image1 = image1.copy()
    annotated_image2 = image2.copy()

    # Annotate unique detections on each image
    for box in unique_to_image1:
        cv2.rectangle(annotated_image1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                      2)  # Red for unique boxes

    for box in unique_to_image2:
        cv2.rectangle(annotated_image2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                      2)  # Red for unique boxes

    # Display the annotated images
    sv.plot_image(annotated_image1, (10, 10))
    sv.plot_image(annotated_image2, (10, 10))


# Example usage
moving_vs_stat("frame_0002.jpg", "frame_0003.jpg")
