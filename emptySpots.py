import numpy as np
import cv2
import matplotlib.pyplot as plt
from supervision.detection.core import Detections
import supervision as sv
from typing import List, Tuple
from segmentation import get_mask_edge_points


def calculate_horizontal_distance(box1, box2):
    """
    Calculate the horizontal distance between two bounding boxes.
    Args:
        box1: (x_left1, y_down1, x_right1, y_up1)
        box2: (x_left2, y_down2, x_right2, y_up2)
    Returns:
        float: The horizontal distance between the edges of the two bounding boxes.
    """
    xmin1, _, xmax1, _ = box1
    xmin2, _, xmax2, _ = box2
    # center1 = (xmin1 + xmax1) / 2
    # center2 = (xmin2 + xmax2) / 2
    # return abs(center1 - center2)
    return (xmin2 - xmax1)


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


def find_minimum_car(detections):
    # Convert detections to a NumPy array for efficient operations
    boxes = np.array(detections)

    # Extract the x-coordinates of the left and right sides
    xl = boxes[:, 0]
    xr = boxes[:, 2]

    # Compute widths
    widths = np.abs(xr - xl)

    # Find the minimum width
    min_width = min(widths)

    return min_width


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


def free_parking_exact_coord(free_spots, free_areas,posture, exact_detections, avg_parking_spot_width):
    """
    Identify free parking spots based on the distance between car detections,
    ensuring no other car overlaps the free spot.

    Arguments:
    - exact_detections: List of arrays, where each array contains four arrays representing the (x, y) coordinates
      of the top-left, top-right, bottom-right, and bottom-left corners of the bounding box.
    - min_parking_spot_width: Minimum width required for a free parking spot.

    Returns:
    - free_spots: List of tuples, each containing two car detections that form the edges of a free parking spot.
    """

    # Sort detections by the x-coordinate of the top-left corner of the bounding boxes
    exact_detections.sort(key=lambda x: x[0])

    for i in range(len(exact_detections) - 1):

        # Calculate the horizontal distance between the current car and the next car
        # d = horizontal_distance_exact_coord(exact_detections[i],
        #                                            exact_detections[i + 1])

        d = calculate_horizontal_distance(exact_detections[i],
                                          exact_detections[i + 1])
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


def display_empty_spot(image, points):
    """
    Display the rectangle formed by four points on the image.
display_empty_spot
    Arguments:
    - image (np.ndarray): The original image.
    - points (List[List[int]]): A list of four points, each represented as [x, y].
    """
    # Ensure there are exactly four points
    if len(points) != 4:
        raise ValueError("There must be exactly four points")

    # Create a copy of the image to draw the rectangle
    img_copy = image.copy()

    # Draw lines between the points to form a rectangle
    cv2.line(img_copy, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
    cv2.line(img_copy, tuple(points[1]), tuple(points[2]), (0, 255, 0), 2)
    cv2.line(img_copy, tuple(points[2]), tuple(points[3]), (0, 255, 0), 2)
    cv2.line(img_copy, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)

    # Draw points for better visibility
    for point in points:
        cv2.circle(img_copy, tuple(point), 5, (255, 0, 0), -1)

    # Display the image with the rectangle
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def detections_in_area(detections, parking_area_bbox):
    xmin_area, ymin_area, xmax_area, ymax_area = parking_area_bbox
    detections_within_area = []
    for detection in detections:  # detections.xyxy in YOLO WORLD
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


def find_empty_spots(image, detections, masks, parking_areas) -> Tuple[List[List[float]], List[str]]:

    free_spots = []
    free_areas = []

    if len(detections) == 0:
        # if there are no cars, the entire area is free
        for parking_area in parking_areas:
            free_spots.append(parking_area[1])
            free_areas.append(parking_area[0])
        return free_spots, free_areas

    for parking_area in parking_areas:
        posture, parking_area_bbox = parking_area
        detections_per_area = detections_in_area(detections, parking_area_bbox)
        # present_results([parking_area_bbox], image)
        # present_results(detections_per_area, image)

        # This is different from the previous condition because it looks in each area
        if len(detections_per_area) == 0:
            free_spots.append(parking_area_bbox)
            free_areas.append(posture)
            continue

        reference_car = find_average_car(detections_per_area)
        # reference_car = find_minimum_car(detections_per_area)
        if posture == 'diagonal':
            exact_coordinates = []
            for mask in masks:
                mask_edge_points = get_mask_edge_points(mask)
                mask_edge_points_list = mask_edge_points.tolist()
                exact_coordinates.append(mask_edge_points_list)
            l = [(int(points[0][0]),
                  int(points[1][1]),
                  int(points[2][0]),
                  int(points[3][1])) for points in exact_coordinates]
            detections_per_area = detections_in_area(l, parking_area_bbox)
            free_parking_exact_coord(free_spots, free_areas, posture, detections_per_area, reference_car)
            free_parking_in_edge(free_spots, free_areas, posture, detections_per_area, reference_car, parking_area_bbox)

        else:
            free_parking_between_cars(free_spots, free_areas, posture, detections_per_area, reference_car)
            free_parking_in_edge(free_spots, free_areas, posture, detections_per_area, reference_car, parking_area_bbox)

    # present_results(free_spots, image)
    return free_spots, free_areas