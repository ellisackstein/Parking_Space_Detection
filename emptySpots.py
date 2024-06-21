import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    #center1 = (xmin1 + xmax1) / 2
    #center2 = (xmin2 + xmax2) / 2
    # return abs(center1 - center2)
    return (xmin2-xmax1)

def edge_horizontal_distance(car,frame,side):
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
        return xmax2 - xmax1
    else:
        return xmin1 - xmin2

def find_smallest_car(detections):
    # Set initial smallest width to positive infinity
    smallest_width = float('inf')
    smallest_height = 0

    boxes = detections.xyxy
    for box in boxes:
        xl, _, xr, _ = box
        width = abs(xr - xl)
        if width < smallest_width:
            smallest_width = width

    return smallest_width


def extract_car_detections(detections):
    """
    Extract bounding boxes of cars from the inference results.
    """
    car_detections = []
    boxes = detections.xyxy
    # the coordinates' order:
    # 1. x_left -> x-coordinate of the left edge
    # 2. y_up -> y-coordinate of the top edge
    # 3. x_right -> x-coordinate of the right edge
    # 4. y_down -> y-coordinate of the bottom edge
    for box in boxes:
        car_detections.append(box)
    return car_detections


def free_parking_between_cars(car_detections, min_parking_spot_width):
    """
    Identify free parking spots based on the distance between car detections,
    ensuring no other car overlaps the free spot.
    """

    # Sort detections by the x-coordinate of the bounding boxes
    car_detections.sort(key=lambda x: x[0])
    free_spots = []

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

        # Check for overlaps with other cars
        overlap = False
        for j in range(len(car_detections)):
            if j != i and j != i + 1:
                car_x1 = car_detections[j][0]
                car_y1 = car_detections[j][1]
                car_x2 = car_detections[j][2]
                car_y2 = car_detections[j][3]

                if not (
                        car_x2 <= x1 or car_x1 >= x2 or car_y2 <= y1 or car_y1 >= y2):
                    overlap = True
                    break

        if not overlap:
            # Calculate the horizontal distance between
            # the current car and the next car
            distance = calculate_horizontal_distance(car_detections[i],
                                                     car_detections[i + 1])
            if distance >= min_parking_spot_width:
                free_spots.append([car_detections[i][2],
                                  min(car_detections[i][1],
                                      car_detections[j][1]),
                                  car_detections[j][0],
                                  max(car_detections[i + 1][3],
                                      car_detections[j][3])])
                # TODO: check if the min and max are correct

    return free_spots


def free_parking_in_edge(car_detections, min_parking_spot_width,parking_area):
    free_spots = []
    if len(car_detections) == 0: # no cars in scene
        return

    car_detections.sort(key=lambda x: x[0])

    # leftest car
    distance1 = edge_horizontal_distance(parking_area,
                                             car_detections[0],0)

    if distance1 >= min_parking_spot_width:
        free_spots.append([parking_area[0],
                           car_detections[1],
                           car_detections[0],
                           car_detections[3]])

    distance2 = edge_horizontal_distance(parking_area,
                                        car_detections[len(car_detections)-1], 1)

    if distance2 >= min_parking_spot_width:
        free_spots.append([car_detections[2],
                           car_detections[1],
                           parking_area[2],
                           car_detections[3]])

    return free_spots


def free_parking_exact_coord(exact_detections, min_parking_spot_width):
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
    exact_detections.sort(key=lambda x: x[0][0])
    free_spots = []

    for i in range(len(exact_detections) - 1):
        # Define the potential free spot area

        top_left_x_1 = exact_detections[i][0][0]  # top-left corner x
        top_left_y_1 = exact_detections[i][0][1]  # top-left corner y

        bottom_right_x_1 = exact_detections[i][1][0]  # bottom-right corner x
        bottom_right_y_1 = exact_detections[i][1][1]  # bottom-right corner y

        bottom_left_x_1 = exact_detections[i][2][0]  # bottom-left corner x
        bottom_left_y_1 = exact_detections[i][2][1]  # bottom-left corner y

        top_right_x_1 = exact_detections[i][3][0]  # top-right corner x
        top_right_y_1 = exact_detections[i][3][1]  # top-right corner y

        # Calculate the horizontal distance between the current car and the next car
        distance = horizontal_distance_exact_coord(exact_detections[i],
                                                   exact_detections[i + 1])
        if distance > min_parking_spot_width:
            free_spots.append(
                [[exact_detections[i][3][0], exact_detections[i][3][1]],
                 # top left
                 [exact_detections[i + 1][2][0], exact_detections[i][2][1]],
                 # bottom right
                 [exact_detections[i][1][0], exact_detections[i][1][1]],
                 # bottom left
                 [exact_detections[i + 1][0][0],
                  exact_detections[i + 1][0][1]]])  # top right

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
    t_r_x1 = box1[3][0]  # top-right x
    b_r_x1 = box1[1][0]  # bottom-right x
    t_l_x2 = box2[0][0]  # top-left x of next car
    b_l_x2 = box2[2][0]  # bottom-left x of next car

    center1 = (t_r_x1 + b_r_x1) / 2
    center2 = (t_l_x2 + b_l_x2) / 2
    return center2 - center1


def display_empty_spot(image, points):
    """
    Display the rectangle formed by four points on the image.

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
