import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from supervision.detection.core import Detections
import supervision as sv
from typing import List, Tuple


def get_mask_edge_points(mask):
    """
    This function calculates the coordinates of the four edge points
    (top-most, bottom-most, left-most, right-most) for each mask.

    Arguments: masks (np.ndarray) - The output masks in CxHxW format, where C is the number of masks,
    and (H, W) is the height and width of the original image.

    Returns: edge_points [np.ndarray] - A list of arrays, each of shape 4x2, where each array contains
    the (x, y) coordinates of the four edge points of a mask.
    """

    if mask.size == 0:
        # If no mask is found, return None for that mask
        return np.array([[None, None], [None, None], [None, None], [None, None]])

    else:
        # Get the edge points
        x_min = mask[np.argmin(mask[:, 0])]
        x_max = mask[np.argmax(mask[:, 0])]
        y_min = mask[np.argmin(mask[:, 1])]
        y_max = mask[np.argmax(mask[:, 1])]
        return np.array([x_min, y_min, x_max, y_max])


def visualize_masks(image_path, masks):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image file '{image_path}'.")
        return

    # Convert image from BGR to RGB (for compatibility with Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw contours for each mask on the image
    for mask in masks:
        # Create a mask image of zeros with same shape as original image
        mask_image = np.zeros_like(image_rgb)

        # Reshape the mask to (n, 1, 2) for cv2.fillPoly
        mask_points = mask.reshape((-1, 1, 2)).astype(np.int32)

        # Draw filled contour for the mask
        cv2.fillPoly(mask_image, [mask_points], (255, 255, 255))  # White color for mask

        # Overlay mask on the original image
        image_rgb = cv2.addWeighted(image_rgb, 1, mask_image, 0.5, 0)

    # Display the image with overlaid masks
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Image with Masks Overlay')
    plt.show()


def calculate_horizontal_distance(box1, box2):
    # This function calculates the horizontal distance between two bounding boxes.
    x_min1, _, x_max1, _ = box1
    x_min2, _, x_max2, _ = box2
    return x_min2 - x_max1


def edge_horizontal_distance(frame, car, side):
    """
    Calculate the horizontal distance between a car and the parking area
    Args:
        car: (x_left1, y_down1, x_right1, y_up1)
        frame: (x_left1, y_down1, x_right1, y_up1)
        side:   1 - car in the left of the frame,
                0 - car in the right of the frame

    Returns:
        float: The horizontal distance
    """
    xmin1, _, xmax1, _ = car
    xmin2, _, xmax2, _ = frame
    if side:
        return xmin1 - xmin2
    else:
        return xmax2 - xmax1


def find_average_car(detections):
    # Convert detections to a NumPy array for efficient operations
    boxes = np.array(detections)

    # Extract the x-coordinates of the left and right sides
    xl = boxes[:, 0]
    xr = boxes[:, 2]

    # Compute widths
    widths = np.abs(xr - xl)

    # Calculate the average width
    average_width = np.mean(widths)

    return average_width


def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def find_average_diagonal_car(coords):
    average_points = []
    for point in coords:
        x1, y1 = point[0]
        x2, y2 = point[1]
        x3, y3 = point[2]
        x4, y4 = point[3]
        avg1, avg2 = [(x1 + x4) / 2, (y1 + y4) / 2], [(x2 + x3) / 2, (y2 + y3) / 2]
        average_points.append(euclidean_distance(avg1, avg2))

    return np.mean(average_points) * 1.25


def free_parking_between_cars(free_spots, free_areas, posture, car_detections, min_parking_spot_width):
    """
    Identify free parking spots based on the distance between car detections,
    ensuring no other car overlaps the free spot.
    """

    # Sort detections by the x-coordinate of the bounding boxes
    car_detections.sort(key=lambda x: x[0])

    for i in range(len(car_detections) - 1):
        # Define the potential free spot area
        # x1: x-coordinate of the right edge of the current car
        x1 = car_detections[i][2]
        # y1: y-coordinate of the top edge of the current car
        y1 = car_detections[i][1]
        # x2: x-coordinate of the left edge of the next car
        x2 = car_detections[i + 1][0]
        # y2: y-coordinate of the bottom edge of the next car
        y2 = car_detections[i + 1][3]

        # Calculate the horizontal distance between
        # the current car and the next car
        distance = calculate_horizontal_distance(car_detections[i],
                                                 car_detections[i + 1])
        if distance >= min_parking_spot_width:
            free_spots.append([car_detections[i][2],
                               min(car_detections[i][1],
                                   car_detections[i + 1][1]),
                               car_detections[i + 1][0],
                               max(car_detections[i + 1][3],
                                   car_detections[i + 1][3])])
            free_areas.append(posture)
            # TODO: check if the min and max are correct

    return free_spots


def free_parking_in_edge(free_spots, free_areas, posture, car_detections, min_parking_spot_width, parking_area):
    car_detections.sort(key=lambda x: x[0])

    # most left car
    left_car = car_detections[0]
    distance1 = edge_horizontal_distance(parking_area, left_car, 1)

    if distance1 >= min_parking_spot_width:
        free_spots.append([parking_area[0],
                           left_car[1],
                           left_car[0],
                           left_car[3]])
        free_areas.append(posture)

    right_car = car_detections[len(car_detections) - 1]
    distance2 = edge_horizontal_distance(parking_area, right_car, 0)

    if distance2 >= min_parking_spot_width:
        free_spots.append([right_car[2],
                           right_car[1],
                           parking_area[2],
                           right_car[3]])
        free_areas.append(posture)

    return free_spots


def free_parking_diagonal_coords(free_spots, free_areas, posture, exact_detections, avg_parking_spot_width):
    """
    This function identifies free parking spots based on the distance between car detections,
    ensuring no other car overlaps the free spot.

    Args:
    exact_detections: List of arrays, where each array contains four arrays representing the (x, y) coordinates
    of the top-left, top-right, bottom-right, and bottom-left corners of the bounding box.
    min_parking_spot_width: Minimum width required for a free parking spot.

    Returns: free_spots
    """

    # Sort detections by the x-coordinate of the top-left corner of the bounding boxes
    exact_detections.sort(key=lambda x: x[0])

    for i in range(len(exact_detections) - 1):

        # Calculate the horizontal distance between the current car and the next car
        d = calculate_horizontal_distance(exact_detections[i], exact_detections[i + 1])
        if d > avg_parking_spot_width:
            free_spots.append(
                [exact_detections[i][2],
                 # top left - min x
                 min(exact_detections[i][1], exact_detections[i + 1][1]),
                 # bottom right - min y
                 exact_detections[i + 1][0],
                 # bottom left - max x
                 max(exact_detections[i][3], exact_detections[i + 1][3])])
            # top right - max y
            free_areas.append(posture)

    return free_spots


def horizontal_distance_exact_coord(box1, box2):
    """
       Calculate the horizontal distance between two bounding boxes.

       Args:
           box1: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
           box2: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
           in that order :
                    [[top_left], [bottom_right], [bottom_left], [top_right]]

       Returns:
           float: The horizontal distance between the centers of the two bounding boxes.
       """
    t_r_x1 = box1[2][0]  # top-right x
    b_r_x1 = box1[3][0]  # bottom-right x
    t_l_x2 = box2[0][0]  # top-left x of next car
    b_l_x2 = box2[1][0]  # bottom-left x of next car

    # center1 = (t_r_x1 + b_r_x1) / 2
    # center2 = (t_l_x2 + b_l_x2) / 2

    top_distance = abs(t_l_x2 - t_r_x1)
    bottom_distance = abs(b_l_x2 - b_r_x1)
    return abs(box2[0][0] - box1[2][0])


def detections_in_area(detections, parking_area_bbox):
    # This function gets a detections list and a parking area bounding box and returns 
    # only the detections that are within the bounding box
    
    xmin_area, ymin_area, xmax_area, ymax_area = parking_area_bbox
    detections_within_area = []
    for detection in detections:  
        xmin_det, ymin_det, xmax_det, ymax_det = detection
        if xmin_area <= int(xmin_det) and \
                ymin_area <= int(ymin_det) and \
                xmax_area >= int(xmax_det) and \
                ymax_area >= int(ymax_det):
            detections_within_area.append(detection)
    return detections_within_area


def present_results(arr, test_path):
    image = cv2.imread(test_path)

    for cord in arr:
        rectangle_coords = cord

        # Draw the rectangle on the image
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (int(rectangle_coords[0]), int(rectangle_coords[1])),
                      (int(rectangle_coords[2]), int(rectangle_coords[3])), color, thickness)

    cv2.namedWindow("Image with Rectangle", cv2.WINDOW_NORMAL)

    # Display the image with the rectangle
    cv2.imshow("Image with Rectangle", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_results(arr, test_path, save_path):
    # This function gets an array (a rectangle) and 2 paths and saves the image with the rectangle on it
    image = cv2.imread(test_path)

    for cord in arr:
        rectangle_coords = cord

        # Draw the rectangle on the image
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (int(rectangle_coords[0]), int(rectangle_coords[1])),
                      (int(rectangle_coords[2]), int(rectangle_coords[3])), color, thickness)

    # Save the image with the rectangles to the specified path
    cv2.imwrite(save_path, image)

    return save_path


def convert_masks_cords(masks_coordinates):
    return [(int(points[0][0]),
             int(points[1][1]),
             int(points[2][0]),
             int(points[3][1])) for points in masks_coordinates]


def handle_diagonal_area(masks, parking_area_bbox, free_spots, free_areas, posture):
    # This function handles the diagonal parking areas

    # Convert multi-dimensional NumPy array to list
    masks_coordinates = [get_mask_edge_points(mask).tolist() for mask in masks]

    # Find average car
    reference_car = find_average_diagonal_car(masks_coordinates)

    # Extract coordinates in the format (min_x, min_y) and (max_x, max_y)
    converted_cords = convert_masks_cords(masks_coordinates)

    # Append the free spaces to free spots array
    detections_per_area = detections_in_area(converted_cords, parking_area_bbox)
    free_parking_diagonal_coords(free_spots, free_areas, posture, detections_per_area, reference_car)
    free_parking_in_edge(free_spots, free_areas, posture, detections_per_area, reference_car, parking_area_bbox)


def handle_horizontal_vertical_area(detections_per_area, free_spots, free_areas, posture, parking_area_bbox):
    # This function handles the horizontal and vertical parking areas

    reference_car = find_average_car(detections_per_area)
    free_parking_between_cars(free_spots, free_areas, posture, detections_per_area, reference_car)
    free_parking_in_edge(free_spots, free_areas, posture, detections_per_area, reference_car, parking_area_bbox)


def find_empty_spots(image_path, detections, masks, parking_areas) -> Tuple[List[List[float]], List[str]]:
    free_spots = []
    free_areas = []

    if len(detections) == 0:
        # If there are no cars at all, the parking areas are free
        for parking_area in parking_areas:
            free_spots.append(parking_area[1])
            free_areas.append(parking_area[0])
        return free_spots, free_areas

    for parking_area in parking_areas:
        posture, parking_area_bbox = parking_area
        detections_per_area = detections_in_area(detections, parking_area_bbox)

        # If there are no cars at a parking area, the parking area is free
        if len(detections_per_area) == 0:
            free_spots.append(parking_area_bbox)
            free_areas.append(posture)
            continue

        if posture == 'diagonal':
            handle_diagonal_area(masks, parking_area_bbox, free_spots, free_areas, posture)

        else:
            handle_horizontal_vertical_area(detections_per_area, free_spots, free_areas, posture, parking_area_bbox)

    return free_spots, free_areas

# present_results([parking_area_bbox], image_path)
# present_results(detections_per_area, image_path)
# present_results(free_spots, image_path)
