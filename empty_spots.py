def calculate_horizontal_distance(box1, box2):
    # Calculate the horizontal distance between two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return abs((x1 + w1 / 2) - (x2 + w2 / 2))


def find_smallest_car(detections):
    # Set initial smallest width to positive infinity
    smallest_width = float('inf')

    boxes = detections.xyxy
    for box in boxes:
        _, _, w, h = box
        if w < smallest_width:
            smallest_width = w
            smallest_height = h

    return smallest_width, smallest_height


def extract_car_detections(detections):
    """
    Extract bounding boxes of cars from the inference results.
    """
    car_detections = []
    boxes = detections.xyxy
    for box in boxes:
        car_detections.append(box)
    return car_detections


def free_parking_between_cars(detections, min_parking_spot_width):
    """
    Identify free parking spots based on the distance between car detections,
    ensuring no other car overlaps the free spot.
    """
    # Sort detections by the x-coordinate of the bounding boxes
    detections.sort(key=lambda x: x[0])
    free_spots = []

    for i in range(len(detections) - 1):
        distance = calculate_horizontal_distance(detections[i],
                                                 detections[i + 1])
        if distance > min_parking_spot_width:
            # Define the potential free spot area
            x1 = int(detections[i][0] + detections[i][2])
            y1 = int(detections[i][1])
            x2 = int(detections[i + 1][0])
            y2 = int(detections[i + 1][1] + detections[i + 1][3])

            # Check for overlaps with other cars
            overlap = False
            for j in range(len(detections)):
                if j != i and j != i + 1:
                    car_x1 = detections[j][0]
                    car_y1 = detections[j][1]
                    car_x2 = detections[j][0] + detections[j][2]
                    car_y2 = detections[j][1] + detections[j][3]

                    if not (
                            car_x2 <= x1 or car_x1 >= x2 or car_y2 <= y1 or car_y1 >= y2):
                        overlap = True
                        break

            if not overlap:
                free_spots.append((detections[i], detections[i + 1]))

    return free_spots
