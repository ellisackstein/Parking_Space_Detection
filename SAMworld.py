from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
from emptySpots import *


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
    - edge_points (List[np.ndarray]): A list of arrays,
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
        top_most = coords[np.argmin(coords[:, 0])]
        bottom_most = coords[np.argmax(coords[:, 0])]
        left_most = coords[np.argmin(coords[:, 1])]
        right_most = coords[np.argmax(coords[:, 1])]
        edge_points_list.append(
            np.array([top_most, bottom_most, left_most, right_most]))

    return edge_points_list


def display_edge_points(image: np.ndarray, edge_points_list: np.ndarray):
    """
    Display the edge points on the image.

    Arguments:
    - image (np.ndarray): The original image.
    - edge_points_list (List[np.ndarray]): A list of arrays, each containing the (x, y) coordinates
      of the four edge points of a mask.
    """
    for edge_points in edge_points_list:
        for point in edge_points:
            if point[0] is not None and point[1] is not None:
                cv2.circle(image, (point[1], point[0]), 5, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


################ SETUP
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                               facecolor=(0, 0, 0, 0), lw=2))


################
SOURCE_IMAGE_PATH = "/Users/shiraadler/PycharmProjects/new/Parking_Space_Detection/test_img/20240426_122224.jpg"
model = YOLOWorld(model_id="yolo_world/l")
classes = ["car"]
model.set_classes(classes)
image = cv2.imread(SOURCE_IMAGE_PATH)

results = model.infer(image)
detections = sv.Detections.from_inference(results)

car_detections = extract_car_detections(detections)

# image = cv2.imread(SOURCE_IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)
predictor.set_image(image)

for box in car_detections:
    # input_box = np.array([425, 600, 700, 875])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False, )

    #mask_edge_points = get_mask_edge_points(masks)
    #print(mask_edge_points)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    #display_edge_points(image, mask_edge_points)
    # show_mask(masks[0], plt.gca())
    # show_box(box, plt.gca())
    # plt.axis('off')

