import os
import xml.etree.ElementTree as ET


def mixed_parking_mark(mixed_test_path):

    # Initialize the list to hold bounding box coordinates
    bboxes = []

    # Check if the scene folder exists
    if os.path.exists(mixed_test_path):
        # Iterate through files in the scene folder
            file = 'parking_area.xml'
            # Full path to the XML file
            xml_file_path = os.path.join(mixed_test_path, file)

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
        print(f"Scene folder {mixed_test_path} does not exist.")

    return bboxes


def parking_mark(scene_num, base_dir):

    # Construct the folder name based on the scene number
    scene_folder = f"scene{scene_num}"

    # Full path to the scene folder
    scene_path = os.path.join(base_dir, scene_folder)

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
