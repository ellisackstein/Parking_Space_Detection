import cv2
import numpy as np
import supervision as sv
from yolo import predict


def find_linear_separator_(detections, annotated_image):

    # Extract the center points of the detections
    centers = []
    for bbox in detections:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        centers.append((center_x, center_y))

    # Convert the centers to a NumPy array
    centers = np.array(centers)

    # Calculate the median y-coordinate
    mean_y = np.mean(centers[:, 1])

    # Prepare to draw the horizontal line on the image
    image_height, image_width, _ = annotated_image.shape
    line_y = int(mean_y)
    line_start = (0, line_y)
    line_end = (image_width, line_y)

    # Draw the horizontal separator line
    cv2.line(annotated_image, line_start, line_end, (0, 0, 255), 8)

    # Define the two parts of the image based on the separator line
    part1 = [0, image_width, line_y, image_height]
    part2 = [0, image_width, 0, line_y]

    # Optionally display the annotated image with the separator
    sv.plot_image(annotated_image, (10, 10)) # Uncomment to display the image

    return [("unknown", [part1]), ("unknown",[part2])]


def find_linear_separator(detections, annotated_image):
    # Extract the center points of the detections
    centers = []
    heights = []
    for bbox in detections:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        height = y2 - y1
        centers.append((center_x, center_y))
        heights.append(height)

    # Convert centers to a NumPy array and calculate mean y-coordinate
    centers = np.array(centers)
    mean_y = np.mean(centers[:, 1])

    # Calculate average height of the bounding boxes
    avg_height = np.mean(heights)

    # Prepare to draw the horizontal line on the image
    image_height, image_width, _ = annotated_image.shape
    line_y = int(mean_y)
    line_start = (0, line_y)
    line_end = (image_width, line_y)

    # Draw the horizontal separator line
    cv2.line(annotated_image, line_start, line_end, (0, 0, 255), 8)

    # Define the two parts of the image based on the separator line
    part1_bboxes = []
    part2_bboxes = []

    # Sort bboxes into upper and lower parts
    for bbox in detections:
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2
        if center_y >= line_y:
            part1_bboxes.append(bbox)
        else:
            part2_bboxes.append(bbox)

    # Calculate y_mean for both parts
    part1_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in part1_bboxes]
    part2_centers = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in part2_bboxes]

    if part1_centers:
        y_mean_upper = np.mean([center[1] for center in part1_centers])
    else:
        y_mean_upper = mean_y  # Fallback if no detections in part1

    if part2_centers:
        y_mean_lower = np.mean([center[1] for center in part2_centers])
    else:
        y_mean_lower = mean_y  # Fallback if no detections in part2

    # Define new areas using y_mean and avg_height
    upper_area_y1 = int(y_mean_upper - avg_height)
    upper_area_y2 = int(y_mean_upper + avg_height)
    lower_area_y1 = int(y_mean_lower - avg_height)
    lower_area_y2 = int(y_mean_lower + avg_height)

    # Bounding boxes of the areas
    upper_area_bbox = [0, upper_area_y1, image_width, upper_area_y2]
    lower_area_bbox = [0, lower_area_y1, image_width, lower_area_y2]

    # Draw bounding boxes on the image
    cv2.rectangle(annotated_image, (0, upper_area_y1), (image_width, upper_area_y2), (0, 255, 0), 2)
    cv2.rectangle(annotated_image, (0, lower_area_y1), (image_width, lower_area_y2), (255, 0, 0), 2)

    # Optionally display the annotated image with the separator and areas
    # sv.plot_image(annotated_image, (10, 10))  # Uncomment to display the image

    return [("upper_area", upper_area_bbox), ("lower_area", lower_area_bbox)]

#
# path = "Tests/empty_spots/scene5/test5/5.png"
# car_boxes,car_masks,image = predict(path)
# find_linear_separator(car_boxes, image)