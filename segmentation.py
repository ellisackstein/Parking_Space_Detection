from segment_anything import SamPredictor, sam_model_registry
import numpy as np

def get_mask_edge_points(masks: np.ndarray):
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
    edge_points_list = []

    for mask in masks:
        # Find the coordinates where the mask is non-zero
        coords = np.column_stack(np.where(mask == 1))
        if coords.size == 0:
            # If no mask is found, return None for that mask
            edge_points_list.append(np.array(
                [[None, None], [None, None], [None, None], [None, None]]))
            continue
        # Get the edge points
        top_left = coords[np.argmin(coords[:, 0])]
        bottom_right = coords[np.argmax(coords[:, 0])]
        bottom_left = coords[np.argmin(coords[:, 1])]
        top_right = coords[np.argmax(coords[:, 1])]
        edge_points_list.append(
            [top_left, bottom_right, bottom_left, top_right])

    return edge_points_list

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

        mask_edge_points = get_mask_edge_points(masks)
        edge_points.append(mask_edge_points)

    return edge_points
