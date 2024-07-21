from segment_anything import SamPredictor, sam_model_registry
import numpy as np


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


def visualize_masks(image_path, masks):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image file '{image_path}'.")
        return

    # Convert image from BGR to RGB (for compatibility with Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw contours for each mask on the image
    for mask in masks:
        # Create a mask image of zeros with same shape as original image
        mask_image = np.zeros_like(image_rgb)

        # Reshape the mask to (n, 1, 2) for cv2.fillPoly
        mask_points = mask.reshape((-1, 1, 2)).astype(np.int32)

        # Draw filled contour for the mask
        cv2.fillPoly(mask_image, [mask_points], (255, 255, 255))  # White color for mask

        # Overlay mask on the original image
        image_rgb = cv2.addWeighted(image_rgb, 1, mask_image, 0.5, 0)

    # Display the image with overlaid masks
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Image with Masks Overlay')
    plt.show()