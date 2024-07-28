from time import sleep

from esp_capture import CameraClient
from preprocessing import preprocessing
from movingVSstat import cancel_moving_cars
from yolo import *
import sys
from parkingAreaIdentification import *
from linearSeparator import find_linear_separator
from emptySpots import *

CONFIGURED = "configured"

def main():
    # Our algorithm gets: path, method (configured, non-configured),
    # and scene ID (for collecting parking zones).

    # http://10.100.102.3
    stream_url = " http://10.100.102.13:81/stream"
    resolution_url = "http://10.100.102.13/control?var=framesize&val=13"
    save_dir = './saved_images'

    client = CameraClient(stream_url, resolution_url, save_dir)

    path = save_dir + "/image_latest.jpg"  # sys.argv[1]
    method = "configured"  # sys.argv[2]
    scene_id = 8  # sys.argv[3]

    # The configured method
    if method == CONFIGURED:
        client.run(1)

        # Step 1 : Get predictions
        detections, masks, annotated_image = predict(path)

        # Step 2 : Distinguishing the parking areas
        parking_areas = parking_mark(scene_id, "Parking_areas")

        # Step 3 : Distinguishing the parking areas
        free_spots, free_areas = find_empty_spots(annotated_image, detections, masks, parking_areas)
        present_results(free_spots, path)

    # The non-configured method
    else:
        client.run(100)

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


if __name__ == '__main__':
    main()