import unittest
import os
import xml.etree.ElementTree as ET
from Utils.Yolo import *
from Utils.LinearSeparator import find_linear_separator
from Utils.ExtractEmptySpots import find_empty_spots


class Tests(unittest.TestCase):
    FILE_NAME = ""
    base_dir = 'EmptySpots'
    parking_area_path = '../ParkingAreas'
    mixed_test_path = 'EmptySpots/mixed'

    def test_scene1_test1(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test12")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test2(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test4(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test5(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test5")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test6(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test6")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test7(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test7")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test8(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test8")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 2
    def test_scene2_test1(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test12")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test2(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test3(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test4(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test5(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test5")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test6(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test6")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test7(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test7")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test8(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test8")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 3
    def test_scene3_test1(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test12")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test2(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test3(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test4(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test5(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test5")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test6(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test6")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test6(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test6")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 4
    def test_scene4_test1(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test12")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene4_test2(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene4_test3(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene4_test4(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 5
    def test_scene5_test1(self):
        scene_path = os.path.join(self.base_dir, "scene5")
        test_path = os.path.join(scene_path, "test12")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene5_test2(self):
        scene_path = os.path.join(self.base_dir, "scene5")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene5_test3(self):
        scene_path = os.path.join(self.base_dir, "scene5")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 6
    def test_scene6_test1(self):
        scene_path = os.path.join(self.base_dir, "scene6")
        test_path = os.path.join(scene_path, "test12")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene6_test2(self):
        scene_path = os.path.join(self.base_dir, "scene6")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene6_test3(self):
        scene_path = os.path.join(self.base_dir, "scene6")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene6_test4(self):
        scene_path = os.path.join(self.base_dir, "scene6")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(test_path)
        for iou in ious:
            self.assertTrue(iou)

    # mixed
    def test_mixed_test1(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test12")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test2(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test1")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test3(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test2")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test4(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test3")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test5(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test5")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test6(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test6")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test7(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test7")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test8(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test8")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test9(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test9")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test10(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test10")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_mixed_test11(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test11")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)


    def calculate_iou(self, box1, box2):
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

    def parse_bounding_boxes(self, xml_file):
        """
        Parse bounding boxes from a Pascal VOC XML annotation file.

        Args: xml_file (str): Path to the XML file.

        Returns: A a bounding box represented as [xmin, ymin, xmax, ymax].
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()

        bbox_list = []
        # Iterate over all object elements
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')

            # Extract coordinates
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # Append coordinates to list
            if (xmin == 0) and (ymin == 0) and (xmax == 0) and (ymax == 0):
                return [[]]
            bbox_list.append([xmin, ymin, xmax, ymax])

        return bbox_list

    def internal_test_code(self, test_path):
        if os.path.isdir(test_path):
            detections, annotated_image = None, None
            png_file, xml_file = None, None

            # Collect PNG and XML files
            for file in os.listdir(test_path):
                if file.endswith('.png'):
                    png_file = os.path.join(test_path, file)
                elif file.endswith('.xml'):
                    xml_file = os.path.join(test_path, file)

            # Process PNG files
            if png_file:
                detections, masks, annotated_image = predict(png_file)

            # Process XML file
            reference_boxes, test_boxes = [], []
            if xml_file:
                # list the correct empty parking spots
                reference_boxes = self.parse_bounding_boxes(xml_file)

                # list the empty parking spots from our algorithm
                parking_areas = find_linear_separator(detections, annotated_image)
                test_boxes = find_empty_spots(png_file, detections, [], parking_areas)

            # self.assertEqual(len(reference_boxes), len(test_boxes))

            ious = []
            for test in test_boxes:
                iou = False
                for reference in reference_boxes:
                    iou_value = self.calculate_iou(test, reference)
                    if iou_value >= 0.7:
                        iou = True
                        break
                ious.append(iou)
            return ious

    def internal_test_mixed_code(self, test_path):
        if os.path.isdir(test_path):
            detections, annotated_image = None, None
            png_file = None
            xml_file = os.path.join(test_path, "empty_spots.xml")

            # Collect PNG and XML files
            for file in os.listdir(test_path):
                if file.endswith('.png') or file.endswith('.jpg'):
                    png_file = os.path.join(test_path, file)

            # Process PNG files
            if png_file:
                detections, masks, annotated_image = predict(png_file)

            # list the correct empty parking spots
            reference_boxes = self.parse_bounding_boxes(xml_file)

            # list the empty parking spots from our algorithm
            parking_areas = find_linear_separator(detections, annotated_image)
            test_boxes = find_empty_spots(png_file, detections, [], parking_areas)

            # self.assertEqual(len(reference_boxes), len(test_boxes))

            ious = []
            for test in test_boxes:
                iou = False
                for reference in reference_boxes:
                    iou_value = self.calculate_iou(test, reference)
                    if iou_value >= 0.7:
                        iou = True
                        break
                ious.append(iou)
            return ious