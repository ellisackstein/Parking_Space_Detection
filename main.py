from preprocessing import preprocessing
from movingVSstat import cancel_moving_cars
from yolo import *
from parkingAreaIdentification import *
from linearSeparator import find_linear_separator
from SAMworld import *  # only for testing and displaying
from emptySpots import *

CONFIGURED = "configured"

if __name__ == '__main__':
    # Our algorithm gets: path, method (configured, non-configured),
    # and scene ID (for collecting parking zones).

    path = sys.argv[1]
    method = sys.argv[2]
    scene_id = sys.argv[3]

    # The configured method
    if method == CONFIGURED:

        # Step 1 : Get predictions
        detections, masks, annotated_image = predict(path)

        # Step 2 : Distinguishing the parking areas
        parking_areas = parking_mark(scene_id, "Parking_areas")

        # Step 3 : Distinguishing the parking areas
        free_spots, free_areas = find_empty_spots(annotated_image, detections, masks, parking_areas)
        present_results(free_spots, path)

    # The non-configured method
    else:
        # Step 1: Preprocessing video of destination
        path1, path2 = preprocessing(path)

        # step 2: Get predictions
        detections1, masks1, annotated_image1 = predict(path1)
        detections2, masks2, annotated_image2 = predict(path2)

        # Step 3: Cancelling moving cars
        detections, masks, annotated_image = cancel_moving_cars(detections1, masks1, annotated_image1,
                                                                detections2, masks2, annotated_image2)

        # Step 4 : if not configured, find a linear separator that defines the parking zones
        parking_areas = find_linear_separator(detections, annotated_image)

        # Step 5: Detecting empty parking spots
        free_spots, free_areas = find_empty_spots(annotated_image, detections, masks, parking_areas)
        present_results(free_spots, path1)
