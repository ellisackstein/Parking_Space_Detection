from Preprocessing import preprocessing
from movingVSstat import cancel_moving_cars
from yolo9 import predict
from parkingAreaIdentification import parking_mark
from LinearSeparator import find_linear_separator

PARKING_AREA = {1: [], 2: [], 3: []}
SUPERVISED = True

# TODO: Elli
# Step 1: Preprocessing video of destination
path_1, path_2 = preprocessing("Scenes/scene1")

# TODO: Elli
# step 2: Get predictions
predictions_1 = predict(path_1)
predictions_2 = predict(path_1)

# TODO: Elli
# Step 3: Cancelling moving cars
final_frame = cancel_moving_cars(path_1, path_2, predictions_1, predictions_2)

# TODO: Shira
# Step 4: Distinguishing the parking areas
if SUPERVISED:
    final_frame = parking_mark(final_frame, PARKING_AREA[1])

else:
    # TODO: Elli
    final_frame = find_linear_separator(final_frame)

# TODO: Shira
# Step 5: Detecting empty parking spots

