# test_path = "Tests/empty_spots/scene2/test1/1.png"
from preprocessing import preprocessing
from movingVSstat import cancel_moving_cars
from yolo import *
from parkingAreaIdentification import *
from linearSeparator import find_linear_separator
from SAMworld import *  # only for testing and displaying
from emptySpots import *

SUPERVISED = True

# Step 1: Preprocessing video of destination
path1, path2 = preprocessing("Scenes\scene1")

# step 2: Get predictions

############ YOLO WORLD ############
# detections1, annotated_image1 = predict(path1)
# detections2, annotated_image2 = predict(path2)
####################################

detections1, masks1, annotated_image1 = predict_yolo_9(path1)
detections2,masks2, annotated_image2 = predict_yolo_9(path2)

# Step 3: Cancelling moving cars
detections,masks, annotated_image = cancel_moving_cars(detections1,masks1, annotated_image1,
                                                 detections2, masks2, annotated_image2)


# Step 4: Distinguishing the parking areas
if SUPERVISED:
    scene_num = 1
    parking_areas = parking_mark(scene_num, "Parking_areas")

else:  # if not supervised, find a linear separator
    parking_areas = find_linear_separator(detections, annotated_image)

# Step 5: Detecting empty parking spots


free_spots = find_empty_spots(annotated_image, detections,masks, parking_areas)

present_results(free_spots, path1)
