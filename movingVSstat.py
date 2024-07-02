import supervision as sv
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

    # calculate top left
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    # calculate bottom right
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
    boxes1 = detections1
    boxes2 = detections2

    unique_to_image1 = []
    unique_to_image2 = boxes2.copy()

    for box1 in boxes1:
        found = False
        for box2 in boxes2:
            iou = calculate_iou(box1, box2)
            if iou >= iou_threshold:
                found = True
                unique_to_image2 = np.array(
                    [box for box in unique_to_image2 if not np.array_equal(box, box2)])
                break
        if not found:
            unique_to_image1.append(box1)

    return unique_to_image1, unique_to_image2


def moving_vs_stat(detections1, annotated_image1, detections2, annotated_image2):
    # Compare detections
    unique_to_image1, unique_to_image2 = compare_detections(detections1, detections2)

    # Annotate unique detections on each image
    for box in unique_to_image1:
        cv2.rectangle(annotated_image1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                      2)  # Red for unique boxes

    for box in unique_to_image2:
        cv2.rectangle(annotated_image2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                      2)  # Red for unique boxes

    # # Display the annotated images
    # sv.plot_image(annotated_image1, (10, 10))
    # sv.plot_image(annotated_image2, (10, 10))

    return unique_to_image1, unique_to_image2


def cancel_moving_cars(detections1, masks1, annotated_image1, detections2,masks2, annotated_image2):
    """
    Determine which set of unique detections has fewer items and return the original detections and annotated image
    corresponding to that set.

    Args:
        unique_to_image1: List of detections unique to image 1.
        annotated_image1: Annotated image corresponding to image 1.
        unique_to_image2: List of detections unique to image 2.
        annotated_image2: Annotated image corresponding to image 2.
        detections1: Original detections object for image 1.
        detections2: Original detections object for image 2.

    Returns:
        original_detections: Original detections object of the image with fewer unique detections.
        original_annotated_image: Annotated image of the image with fewer unique detections.
    """
    unique_to_image1, unique_to_image2 = moving_vs_stat(detections1, annotated_image1, detections2, annotated_image2)
    if len(unique_to_image1) <= len(unique_to_image2):
        print("frame 1!")
        return detections1, masks1,annotated_image1
    else:
        print("frame 2!")
        return detections2, masks2,annotated_image2

# import os
# import cv2
# # import numpy as np
# # import supervision as sv
# from inference.models.yolo_world.yolo_world import YOLOWorld
#
# def calculate_iou(box1, box2):
#     """
#     Calculate the Intersection over Union (IoU) of two bounding boxes.
#     Args:
#         box1, box2: Bounding boxes in the format [x1, y1, x2, y2]
#     Returns:
#         iou: IoU value
#     """
#     x1_inter = max(box1[0], box2[0])
#     y1_inter = max(box1[1], box2[1])
#     x2_inter = min(box1[2], box2[2])
#     y2_inter = min(box1[3], box2[3])
#
#     inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
#
#     box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
#     box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
#
#     iou = inter_area / float(box1_area + box2_area - inter_area)
#     return iou
#
#
# def compare_detections(detections1, detections2, iou_threshold=0.70):
#     """
#     Compare two sets of detections and find unique detections in each set based on IoU.
#     Args:
#         detections1, detections2: Detections objects with `xyxy` attribute containing bounding boxes.
#         iou_threshold: IoU threshold to consider two boxes as the same.
#     Returns:
#         unique_to_image1: List of detections unique to image 1.
#         unique_to_image2: List of detections unique to image 2.
#     """
#     boxes1 = detections1.xyxy
#     boxes2 = detections2.xyxy
#
#     unique_to_image1 = []
#     unique_to_image2 = boxes2.copy()
#
#     for box1 in boxes1:
#         found = False
#         for box2 in boxes2:
#             iou = calculate_iou(box1, box2)
#             if iou >= iou_threshold:
#                 found = True
#                 unique_to_image2 = np.array([box for box in unique_to_image2 if not np.array_equal(box, box2)])
#                 break
#         if not found:
#             unique_to_image1.append(box1)
#
#     return unique_to_image1, unique_to_image2
#
#
# def moving_vs_stat(image1_path, image2_path, model):
#     # Process the first image
#     image1 = cv2.imread(image1_path)
#     results1 = model.infer(image1)
#     detections1 = sv.Detections.from_inference(results1)
#
#     # Process the second image
#     image2 = cv2.imread(image2_path)
#     results2 = model.infer(image2)
#     detections2 = sv.Detections.from_inference(results2)
#
#     # Compare detections
#     unique_to_image1, unique_to_image2 = compare_detections(detections1, detections2)
#
#     return unique_to_image1, unique_to_image2
#
#
# def cancel_moving_cars(directory_path):
#     # Load the model and set classes
#     model = YOLOWorld(model_id="yolo_world/l")
#     classes = ["car"]
#     model.set_classes(classes)
#
#     # Get the last 6 frames in the folder
#     frames = sorted(
#         [f for f in os.listdir(directory_path) if f.endswith(".jpg")],
#         key=lambda x: os.path.getmtime(os.path.join(directory_path, x)))
#     frames = frames[-6:]
#
#     if len(frames) < 2:
#         print("Not enough frames to compare.")
#         return None
#
#     # Compare each pair of frames
#     reference_frame_path = os.path.join(directory_path, frames[0])
#     for frame in frames[1:]:
#         current_frame_path = os.path.join(directory_path, frame)
#         unique_to_image1, unique_to_image2 = moving_vs_stat(reference_frame_path, current_frame_path, model)
#
#         # If the reference frame has no unique detections, use the current frame as the new reference
#         if not unique_to_image1:
#             reference_frame_path = current_frame_path
#
#     return reference_frame_path
#
#
# # Example usage
# directory_path = "Scenes\scene1\VIDEO_20240523_105751358"
# best_frame = cancel_moving_cars(directory_path)
# if best_frame:
#     print(f"The frame with no moving cars is: {best_frame}")
#     best_frame_image = cv2.imread(best_frame)
#     sv.plot_image(best_frame_image, (10, 10))
