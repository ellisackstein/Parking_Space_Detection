from preprocessing import preprocessing
from movingVSstat import cancel_moving_cars
from yolo import predict
from parkingAreaIdentification import *
from linearSeparator import find_linear_separator
from segmentation import *
from SAMworld import *  # only for testing and displaying

SUPERVISED = True

# Step 1: Preprocessing video of destination
# path1, path2 = preprocessing("Scenes\scene1")
# path1 = "test_img/yas.jpg"
# path2 = "test_img/yas.jpg"
#
# # step 2: Get predictions
# detections1, annotated_image1 = predict(path1)
# detections2, annotated_image2 = predict(path2)
#
# # Step 3: Cancelling moving cars
# detections, annotated_image = cancel_moving_cars(detections1, annotated_image1,
#                                                  detections2, annotated_image2)
#


path = "test_img/yas.jpg"
img = cv2.imread(path)
detections, annotated_image = predict(path)
smallest_car_in_scene = find_smallest_car(detections)
car_detections = extract_car_detections(detections)

# # Step 4: Distinguishing the parking areas
if SUPERVISED:
    scene_num = 1
    parking_areas = parking_mark(scene_num)

    for parking_area in parking_areas:
        posture = parking_area[0]
    parking_area_bbox = parking_area[1]

    detections_in_area = detection_in_area(car_detections, parking_area_bbox)

    if posture == 'diagonal':  # TODO: Ellie to change to currect format

        image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        exact_detections = get_edge_points(detections_in_area, image_rgb)

        # displaying the points
        for cord in exact_detections:
            display_edge_points(annotated_image, cord)

            # free_parking_between_cars(exact_detections, smallest_car_in_scene)
        free_spots = free_parking_exact_coord(exact_detections,
                                              smallest_car_in_scene)

        display_empty_spot(annotated_image, free_spots)


    else:
        free_spots = free_parking_between_cars(car_detections,
                                               smallest_car_in_scene)

        # x1,x2,y1,y2 = free_spots[0]
        # image = img.copy()
        # cv2.rectangle(image, (x1,y1),
        #               (x2, y2),
        #               (0, 255, 0), 2)
        # sv.plot_image(image, (10, 10))

# else:
# annotated_image = find_linear_separator(detections, annotated_image)

# TODO: Shira
# Step 5: Detecting empty parking spots




