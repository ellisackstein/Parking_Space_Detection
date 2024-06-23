from preprocessing import preprocessing
from movingVSstat import cancel_moving_cars
from yolo import predict
from parkingAreaIdentification import *
from linearSeparator import find_linear_separator
from segmentation import *
from SAMworld import *  # only for testing and displaying
from supervision.detection.core import Detections

SUPERVISED = True

# Step 1: Preprocessing video of destination
# path1, path2 = preprocessing("Scenes\scene1")

# step 2: Get predictions
test_path = "Tests/empty_spots/scene2/test2/2.png"
detections1, annotated_image1 = predict(test_path)
# detections2, annotated_image2 = predict(path2)

# Step 3: Cancelling moving cars
# detections, annotated_image = cancel_moving_cars(detections1, annotated_image1,
#                                                  detections2, annotated_image2)
detections, annotated_image = detections1, annotated_image1
# Step 4: Distinguishing the parking areas

if SUPERVISED:
    scene_num = 1
    parking_areas = parking_mark(scene_num)

else:  # if not supervised, find a linear separator
    parking_areas = find_linear_separator(detections, annotated_image)

# Step 5: Detecting empty parking spots
reference_car = find_smallest_car(detections)
free_spots = []
for parking_area in parking_areas:
    posture, parking_area_bbox = parking_area
    detections_per_area = detections_in_area(detections, parking_area_bbox)

    detections_per_area_ = Detections(xyxy=np.array(detections_per_area).reshape(len(detections_per_area), 4),
                                     class_id=np.array([1 for _ in range(len(detections_per_area))]))
    BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
    LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
    annotated_image = annotated_image.copy()
    annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections_per_area_)
    sv.plot_image(annotated_image, (10, 10))

    if posture == 'diagonal':
        # image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        # exact_detections = get_edge_points(detections_per_area, image_rgb)
        # # displaying the points
        # for cord in exact_detections:
        #     display_edge_points(annotated_image, cord)
        # free_spots = free_parking_exact_coord(exact_detections, reference_car)
        # display_empty_spot(annotated_image, free_spots)
        continue
    else:
        free_spots = free_parking_between_cars(detections_per_area,reference_car)
        free_spots.append(free_parking_in_edge(detections_per_area,reference_car,parking_area_bbox))
        # if there arent any free spots, free_spots = [None]
        print(free_spots)


arr = free_spots[0]
image = cv2.imread(test_path)
for cord in arr:
    rectangle_coords = cord

    # Draw the rectangle on the image
    color = (0, 255, 0)  # Green color (BGR format)
    thickness = 2        # Line thickness
    cv2.rectangle(image, (int(rectangle_coords[0]), int(rectangle_coords[1])),
                  (int(rectangle_coords[2]), int(rectangle_coords[3])), color, thickness)

# Display the image with the rectangle
cv2.imshow("Image with Rectangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
