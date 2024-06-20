def calculate_horizontal_distance(box1, box2):
    """
    Calculate the horizontal distance between two bounding boxes.
    Args:
        box1: (x_left1, y_up1, x_right1, y_down1)
        box2: (x_left2, y_up2, x_right2, y_down2)
    Returns:
        float: The horizontal distance between the centers of the two bounding boxes.
    """
    xl1, _, xr1, _ = box1
    xl2, _, xr2, _ = box2
    center1 = (xl1 + xr1) / 2
    center2 = (xl2 + xr2) / 2
    return abs(center1 - center2)

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
            # Calculate the horizontal distance between the current car and the next car
            distance = calculate_horizontal_distance(car_detections[i],
                                                     car_detections[i + 1])
            if distance > min_parking_spot_width:
                free_spots.append((car_detections[i], car_detections[i + 1]))

    return free_spots


def free_parking_exact_coord(exact_detections, min_parking_spot_width):
    """
    Identify free parking spots based on the distance between car detections,
    ensuring no other car overlaps the free spot.
    """

    # Sort detections by the x-coordinate of the bounding boxes
    exact_detections.sort(key=lambda x: x[0][0])
    free_spots = []

    for i in range(len(exact_detections) - 1):
        # Define the potential free spot area
        # x1: x-coordinate of the right edge of the current car
        x1 = exact_detections[i][1][0]
        # y1: y-coordinate of the top edge of the current car
        y1 = exact_detections[i][0][1]
        # x2: x-coordinate of the left edge of the next car
        x2 = exact_detections[i + 1][0][0]
        # y2: y-coordinate of the bottom edge of the next car
        y2 = exact_detections[i + 1][2][1]

        # Check for overlaps with other cars
        overlap = False
        for j in range(len(exact_detections)):
            if j != i and j != i + 1:
                car_x1 = exact_detections[j][0][0]  # top-left corner x
                car_y1 = exact_detections[j][0][1]  # top-left corner y
                car_x2 = exact_detections[j][1][0]  # top-right corner x
                car_y2 = exact_detections[j][2][1]  # bottom-right corner y

                if not (
                        car_x2 <= x1 or car_x1 >= x2 or car_y2 <= y1 or car_y1 >= y2):
                    overlap = True
                    break

        if not overlap:
            # Calculate the horizontal distance between the current car and the next car
            distance = horizontal_distance_exact_coord(exact_detections[i],
                                                     exact_detections[i + 1])
            if distance > min_parking_spot_width:
                free_spots.append((exact_detections[i], exact_detections[i + 1]))

    return free_spots


def horizontal_distance_exact_coord(box1, box2):
    """
       Calculate the horizontal distance between two bounding boxes.

       Args:
           box1: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
           box2: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

       Returns:
           float: The horizontal distance between the centers of the two bounding boxes.
       """
    xl1 = box1[0][0]  # top-left x
    xr1 = box1[1][0]  # top-right x
    xl2 = box2[0][0]  # top-left x of next car
    xr2 = box2[1][0]  # top-right x of next car

    center1 = (xl1 + xr1) / 2
    center2 = (xl2 + xr2) / 2
    return abs(center1 - center2)