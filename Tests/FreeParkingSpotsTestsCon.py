import unittest
import os
import xml.etree.ElementTree as ET

import emptySpots
from yolo import *
from parkingAreaIdentification import parking_mark, mixed_parking_mark
from emptySpots import find_empty_spots
import matplotlib.pyplot as plt


class Tests(unittest.TestCase):
    FILE_NAME = ""
    base_dir = 'empty_spots'
    parking_area_path = '../Parking_areas'
    mixed_test_path = '../Tests/empty_spots/mixed'
    success_dict = {"parallel": 0, "vertical": 0, "diagonal": 0}
    failure_dict = {"parallel": 0, "vertical": 0, "diagonal": 0}

    def test_scene1_test1(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test2(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test4(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test4")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test5(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test5")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test6(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test6")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test7(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test7")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene1_test8(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test8")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 2
    def test_scene2_test1(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test2(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test3(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test4(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test5(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test5")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test6(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test6")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test7(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test7")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene2_test8(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test8")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 3
    def test_scene3_test1(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test2(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test3(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test4(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test5(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test5")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test6(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test6")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 4
    def test_scene4_test1(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene4_test2(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene4_test3(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene4_test4(self):
        # this test doesn't pass because there is a motorcycle in the parking,
        # and we don't detect anything except cars
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test4")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 5
    def test_scene5_test1(self):
        scene_path = os.path.join(self.base_dir, "scene5")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene5_test2(self):
        scene_path = os.path.join(self.base_dir, "scene5")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene5_test3(self):
        scene_path = os.path.join(self.base_dir, "scene5")
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 6
    def test_scene6_test1(self):
        scene_path = os.path.join(self.base_dir, "scene6")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene6_test2(self):
        scene_path = os.path.join(self.base_dir, "scene6")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene6_test3(self):
        scene_path = os.path.join(self.base_dir, "scene6")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # Scene 7
    def test_scene7_test1(self):
        scene_path = os.path.join(self.base_dir, "scene7")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene7_test2(self):
        scene_path = os.path.join(self.base_dir, "scene7")
        test_path = os.path.join(scene_path, "test1")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene7_test3(self):
        scene_path = os.path.join(self.base_dir, "scene7")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene7_test4(self):
        # this test doesn't pass because there is a motorcycle in the parking,
        # and we don't detect anything except cars
        scene_path = os.path.join(self.base_dir, "scene7")
        test_path = os.path.join(scene_path, "test4")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene7_test5(self):
        scene_path = os.path.join(self.base_dir, "scene7")
        test_path = os.path.join(scene_path, "test5")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # mixed
    def test_mixed_test1(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test1")
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

    def test_mixed_test12(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test12")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene6_test13(self):
        mixed_test_path = os.path.join(self.mixed_test_path, "test13")
        ious = self.internal_test_mixed_code(mixed_test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_success_failure(self):
        # Categories
        categories = ['parallel', 'vertical', 'diagonal']

        # Values
        success_values = [self.success_dict[cat] for cat in categories]
        failure_values = [self.failure_dict[cat] for cat in categories]

        # Bar width
        bar_width = 0.4

        # Bar positions
        r1 = range(len(categories))
        r2 = [x + bar_width for x in r1]

        # Plotting bars
        plt.bar(r1, success_values, color='green', width=bar_width, edgecolor='grey', label='Success')
        plt.bar(r2, failure_values, color='red', width=bar_width, edgecolor='grey', label='Failure')

        # Adding labels
        plt.xlabel('Category', fontweight='bold')
        plt.xticks([r + bar_width / 2 for r in range(len(categories))], categories)
        plt.ylabel('Count', fontweight='bold')

        # Adding legend
        plt.legend()

        # Setting y-ticks to all integers from 0 to max value + 1
        max_value = max(max(success_values), max(failure_values))
        plt.yticks(range(0, max_value + 2))

        # Display plot
        plt.show()
        self.assertTrue(True)

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

    def internal_test_code(self, scene_path, test_path):
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
            reference_boxes, test_boxes, test_areas = [], [], []
            if xml_file:
                # list the correct empty parking spots
                reference_boxes = self.parse_bounding_boxes(xml_file)

                # list the empty parking spots from our algorithm
                parking_areas = parking_mark(int(scene_path[-1]), self.parking_area_path)
                test_boxes, test_areas = find_empty_spots(png_file, detections, masks, parking_areas)

            equal_len = (len(reference_boxes) == len(test_boxes))

            ious = []
            for i in range(len(test_boxes)):
                iou = False
                for reference in reference_boxes:
                    iou_value = self.calculate_iou(test_boxes[i], reference)
                    if iou_value >= 0.7:
                        iou = True
                        break
                ious.append(iou)

                # add details for graph
                if iou == True:
                    self.success_dict[test_areas[i]] += 1
                else:
                    self.failure_dict[test_areas[i]] += 1

            ious.append(equal_len)
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
            parking_areas = mixed_parking_mark(test_path)
            test_boxes, test_areas = find_empty_spots(png_file, detections, masks, parking_areas)

            equal_len = (len(reference_boxes) == len(test_boxes))

            ious = []
            for i in range(len(test_boxes)):
                iou = False
                for reference in reference_boxes:
                    iou_value = self.calculate_iou(test_boxes[i], reference)
                    if iou_value >= 0.7:
                        iou = True
                        break
                ious.append(iou)
                # add details for graph
                if iou == True:
                    self.success_dict[test_areas[i]] += 1
                else:
                    self.failure_dict[test_areas[i]] += 1

            ious.append(equal_len)
            return ious
