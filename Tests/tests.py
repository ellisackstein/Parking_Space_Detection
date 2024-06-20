import xml.etree.ElementTree as ET
import os
from yolo import predict
from movingVSstat import moving_vs_stat

def parse_bounding_boxes(xml_file):
    """
    Parse bounding boxes from a Pascal VOC XML annotation file.

    Args: xml_file (str): Path to the XML file.

    Returns: A a bounding box represented as [xmin, ymin, xmax, ymax].
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # List to hold bounding boxes
    bounding_boxes = []
    xmin, ymin, xmax, ymax = 0, 0, 0, 0

    # Iterate over all object elements
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')

        # Extract coordinates
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Append coordinates to list
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        box1, box2: Bounding boxes in the format [x1, y1, x2, y2]
    Returns:
        iou: IoU value
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def process_test_directories(base_dir):
    """
    Process each test directory and compare bounding boxes using IoU.
    Args:
        base_dir (str): Path to the base directory containing test directories.
    """
    for test_dir in os.listdir(base_dir):
        test_path = os.path.join(base_dir, test_dir)
        if os.path.isdir(test_path):
            detections = [None, None]
            annotated_images = [None, None]
            png_files = []
            xml_file = None

            # Collect PNG and XML files
            for file in os.listdir(test_path):
                if file.endswith('.png'):
                    png_files.append(os.path.join(test_path, file))
                elif file.endswith('.xml'):
                    xml_file = os.path.join(test_path, file)

            # Ensure there are exactly 2 PNG files
            if len(png_files) != 2:
                print(f"Expected 2 PNG files in {test_path}, but found {len(png_files)}.")
                continue

            # Process PNG files
            for i, png_file in enumerate(sorted(png_files)):
                detections[i], annotated_images[i] = predict(png_file)

            # Process XML file
            if xml_file:
                reference_box = cancel_moving_cars(detections[0], annotated_images[0], detections[1], annotated_images[1])
                test_box = parse_bounding_boxes(xml_file)
                if len(reference_box) == 0:
                    print("FAIL")
                    continue
                iou_value = calculate_iou(test_box, reference_box[0])
                if iou_value >= 0.7:
                    print(f'IoU values for {test_box}: {iou_value} - SUCCESS')
                if iou_value < 0.7:
                    print(f'IoU values for {test_box}: {iou_value} - FAIL')


def cancel_moving_cars(detections1, annotated_image1, detections2, annotated_image2):
    """
    Determine which set of unique detections has fewer items and return the original detections and annotated image
    corresponding to that set.

    Args:
        unique_to_image1: List of detections unique to image 1.
        annotated_image1: Annotated image corresponding to image 1.
        unique_to_image2: List of detections unique to image 2.
        annotated_image2: Annotated image corresponding to image 2.
        detections1: Original detections object for image 1.
        detections2: Original detections object for image 2.

    Returns:
        original_detections: Original detections object of the image with fewer unique detections.
        original_annotated_image: Annotated image of the image with fewer unique detections.
    """

    unique_to_image1, unique_to_image2 = moving_vs_stat(detections1, annotated_image1, detections2, annotated_image2)
    if len(unique_to_image1) > len(unique_to_image2):
        print("frame 1!")
        return unique_to_image1
    else:
        print("frame 2!")
        return unique_to_image2


# Define your base directory containing test directories
base_directory = '../Tests/Scene2'

# Run the processing function
process_test_directories(base_directory)