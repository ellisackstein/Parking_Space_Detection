from Utils.EspCapture import CameraClient
from Utils.Preprocessing import preprocessing
from Utils.CancelMovingCars import cancel_moving_cars
from Utils.Yolo import *
from Utils.MarkParkingArea import *
from Utils.LinearSeparator import find_linear_separator
from Utils.ExtractEmptySpots import *

CONFIGURED = "configured"
ADDRESSES = {1: "", 2: "", 3: "", 4: "", 5: "", 6: "", 7: "", 8: "Harav Hen 10, Jerusalem, Israel"}


def run():
    # Our algorithm gets: path, method (configured, non-configured),
    # and scene ID (for collecting parking zones).

    stream_url = "http://93.172.14.170:91/stream"
    resolution_url = "http://93.172.14.170:90/control?var=framesize&val=13"
    save_dir = 'LiveStreamImages'

    client = CameraClient(stream_url, resolution_url, save_dir)

    path = save_dir + "/image_latest.jpg"
    method = "configured"
    scene_id = 8

    # The configured method
    if method == CONFIGURED:
        client.run(1)

        # Step 1 : Get predictions
        detections, masks = predict(path)

        # Step 2 : Distinguishing the parking areas
        parking_areas = parking_mark(scene_id, "ParkingAreas")

        # Step 3 : Distinguishing the parking areas
        free_spots, free_areas = find_empty_spots(path, detections, masks, parking_areas)
        save_results(free_spots, path, 'HaniApp/static/res/image_latest.jpg')

        # step 4 : return the empty spot
        if len(free_spots) != 0:
            return {ADDRESSES[8]: 'static/res/image_latest.jpg'}
        return {}

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
    run()