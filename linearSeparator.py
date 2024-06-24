import cv2
import numpy as np
import supervision as sv


def find_linear_separator(detections, annotated_image):

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
