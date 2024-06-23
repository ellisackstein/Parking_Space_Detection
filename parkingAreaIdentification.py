import os
import xml.etree.ElementTree as ET


def parking_mark(scene_num):
    # Define the path to the "Scenes" directory
    base_dir = "Scenes"

    # Construct the folder name based on the scene number
    scene_folder = f"scene{scene_num}"

    # Full path to the scene folder
    scene_path = os.path.join(base_dir, scene_folder, "parking_area")

    # Initialize the list to hold bounding box coordinates
    bboxes = []

    # Check if the scene folder exists
    if os.path.exists(scene_path) and os.path.isdir(scene_path):
        # Iterate through files in the scene folder
        for file in os.listdir(scene_path):
            if file.endswith(".xml"):  # Look for XML files
                # Full path to the XML file
                xml_file_path = os.path.join(scene_path, file)

                # Parse the XML file
                tree = ET.parse(xml_file_path)
                root = tree.getroot()

                # Iterate through each object in the XML
                for obj in root.findall("object"):
                    # Extract posture
                    posture = obj.find("name").text
                    # Get the bounding box element
                    bndbox = obj.find("bndbox")
                    # Extract coordinates
                    xmin = int(bndbox.find("xmin").text)
                    ymin = int(bndbox.find("ymin").text)
                    xmax = int(bndbox.find("xmax").text)
                    ymax = int(bndbox.find("ymax").text)

                    # Append the coordinates as a list to bboxes
                    bboxes.append((posture, [xmin, ymin, xmax, ymax]))
    else:
        print(f"Scene folder {scene_folder} does not exist.")

    return bboxes


def detections_in_area(detections, parking_area_bbox):
    xmin_area, ymin_area, xmax_area, ymax_area = parking_area_bbox
    detections_within_area = []
    for detection in detections:
        xmin_det, ymin_det, xmax_det, ymax_det = detection
        if      xmin_area <= xmin_det and \
                ymin_area <= ymin_det and \
                xmax_area >= xmax_det and\
                ymax_area >= ymax_det:
            detections_within_area.append(detection)

    return detections_within_area
