from Preprocessing import preprocessing
from movingVSstat import cancel_moving_cars
from yolo import predict
from parkingAreaIdentification import parking_mark
from linearSeparator import find_linear_separator
from segmentation import *
from SAMworld import *  # only for testing and displaying

SUPERVISED = True

# Step 1: Preprocessing video of destination
#path1, path2 = preprocessing("Scenes\scene1")
path1 = "test_img/amen.jpg"
path2 = "test_img/amen.jpg"

# step 2: Get predictions
detections1, annotated_image1 = predict(path1)
detections2, annotated_image2 = predict(path2)

# Step 3: Cancelling moving cars
detections, annotated_image = cancel_moving_cars(detections1, annotated_image1,
                                                 detections2, annotated_image2)

# Step 4: Distinguishing the parking areas
if SUPERVISED:
    scene_num = 1
    parking_areas = parking_mark(scene_num)

else:
    annotated_image = find_linear_separator(detections, annotated_image)

# TODO: Shira
# Step 5: Detecting empty parking spots

smallest_car_in_scene = find_smallest_car(detections)

car_detections = extract_car_detections(detections)

image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
exact_detections = get_edge_points(car_detections, image)

    # displaying the points
for cord in exact_detections:
    display_edge_points(image, cord)

    #free_parking_between_cars(exact_detections, smallest_car_in_scene)
print(free_parking_exact_coord(exact_detections,smallest_car_in_scene))


