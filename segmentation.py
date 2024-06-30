from segment_anything import SamPredictor, sam_model_registry
import numpy as np
from yolo import *
from SAMworld import *

def get_mask_edge_points(mask):
    """
    Get the coordinates of the four edge points
    (top-most, bottom-most, left-most, right-most)
    for each mask.

    Arguments:
    - masks (np.ndarray): The output masks in CxHxW format,
      where C is the number of masks,
      and (H, W) is the height and width of the original image.

    Returns:
    - edge_points [np.ndarray]: A list of arrays,
      each of shape 4x2, where each array contains the (x, y) coordinates
      of the four edge points of a mask.
    """
    # Find the coordinates where the mask is non-zero
    #coords = np.column_stack(np.where(mask == 1))

    # YOLOV9 - already in array
    coords = mask

    if coords.size == 0:
        # If no mask is found, return None for that mask
        return np.array([[None, None], [None, None], [None, None], [None, None]])
    else:
        # Get the edge points
        x_min = coords[np.argmin(coords[:, 0])]
        x_max = coords[np.argmax(coords[:, 0])]
        y_min = coords[np.argmin(coords[:, 1])]
        y_max = coords[np.argmax(coords[:, 1])]
        return np.array([x_min, y_min, x_max, y_max])

def get_edge_points(detections, annotated_image):
    image = annotated_image
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    edge_points = []

    for box in detections:
        # input_box = np.array([425, 600, 700, 875])
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False, )

        mask_edge_points = get_mask_edge_points(masks[0])
        edge_points.append(mask_edge_points)

    return edge_points

##################### TESTING #####################

# img_path = "test_img/20240426_111900.jpg"
#
# car_boxes, car_masks = predict_yolo_9(img_path)
# image = cv2.imread(img_path)
#
# exact_coordinates = []
# for mask in car_masks:
#     mask_edge_points = get_mask_edge_points(mask)
#     exact_coordinates.append(mask_edge_points)
#     print(mask_edge_points)
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     display_edge_points(image, mask_edge_points)
#     show_mask(mask, plt.gca())
#     plt.axis('off')
#
# free_spots = []
# reference_car = find_average_car(exact_coordinates)
# free_spots = free_parking_exact_coord(free_spots,exact_coordinates,300)
# present_results(free_spots,img_path)

###################################################