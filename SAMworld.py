from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
from emptySpots import *

################ SETUP
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
               s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green',
                               facecolor=(0, 0, 0, 0), lw=2))

################
SOURCE_IMAGE_PATH = "test_img/WhatsApp Image 2024-04-25 at 13.22.14.jpeg"
model = YOLOWorld(model_id="yolo_world/l")
classes = ["car"]
model.set_classes(classes)
image = cv2.imread(SOURCE_IMAGE_PATH)

results = model.infer(image)
detections = sv.Detections.from_inference(results)

car_detections = extract_car_detections (detections)

#image = cv2.imread(SOURCE_IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"


sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#sam.to(device = device)
predictor = SamPredictor(sam)
predictor.set_image(image)

for box in car_detections:

#input_box = np.array([425, 600, 700, 875])
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False,)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(box, plt.gca())
    #show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()