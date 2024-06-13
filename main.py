from Preprocessing import preprocessing
from movingVSstat import cancel_moving_cars
from yolo import predict
from parkingAreaIdentification import parking_mark
from LinearSeparator import find_linear_separator

PARKING_AREA = {1: [], 2: [], 3: []}
SUPERVISED = False


# Step 1: Preprocessing video of destination
path1, path2 = preprocessing("Scenes\scene1")
# path1 = "test_img\Screenshot 2024-06-03 122521.png"
# path2 = "test_img\Screenshot 2024-06-03 122621.png"

# step 2: Get predictions
detections1, annotated_image1 = predict(path1)
detections2, annotated_image2 = predict(path2)

# Step 3: Cancelling moving cars
detections, annotated_image = cancel_moving_cars(detections1, annotated_image1, detections2, annotated_image2)

# Step 4: Distinguishing the parking areas
if SUPERVISED:
    # TODO: Shira
    annotated_image = parking_mark(detections, annotated_image, PARKING_AREA[1])

else:
    annotated_image = find_linear_separator(detections, annotated_image)

# TODO: Shira
# Step 5: Detecting empty parking spots