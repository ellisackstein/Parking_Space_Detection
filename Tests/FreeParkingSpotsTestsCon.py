import unittest
import os
import xml.etree.ElementTree as ET

import emptySpots
from yolo import *
from parkingAreaIdentification import parking_mark, mixed_parking_mark
from emptySpots import find_empty_spots, detections_in_area
import matplotlib.pyplot as plt


class Tests(unittest.TestCase):
    FILE_NAME = ""
    base_dir = 'empty_spots'
    parking_area_path = '../Parking_areas'
    mixed_test_path = '../Tests/empty_spots/mixed'
    ious_values = []
    delta_lens = []
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
        test_path = os.path.join(scene_path, "test2")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test3(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test3")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    def test_scene3_test4(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test4")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # One of the spots is detected successfully.
    # Another one is detected because YOLO couldn't detect a car behind the tree.
    def test_scene3_test5(self):
        scene_path = os.path.join(self.base_dir, "scene3")
        test_path = os.path.join(scene_path, "test5")
        ious = self.internal_test_code(scene_path, test_path)
        for iou in ious:
            self.assertTrue(iou)

    # One of the spots is detected successfully.
    # Another one is detected because YOLO couldn't detect a car behind the tree.
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
        test_path = os.path.join(scene_path, "test2")
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
        #failed because avg car size bigger than free space in right edge
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
        # 1. yolo doesn't detect the car in the middle , so we get a free spot
        # 2. the minimum size of diagonal car doesn't fit the free spot in between cars
        # 3. the size in the top right doesn't fit the free spot on the right
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

    @classmethod
    def plot_success_failure_bars(cls):
        # Categories
        categories = ['parallel', 'vertical', 'diagonal']

        # Values
        success_values = [cls.success_dict[cat] for cat in categories]
        failure_values = [cls.failure_dict[cat] for cat in categories]

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

    @classmethod
    def plot_ious_bar(cls):
        # Generate the x array
        x = list(range(1, len(cls.ious_values) + 1))

        # y array is the ious_value array itself
        y = cls.ious_values

        # Create the bar graph
        plt.bar(x, y)

        # Add labels and title
        plt.xlabel('Index')
        plt.ylabel('IoU Value')
        plt.title('IoU Values Bar Graph')

        # Show the plot
        plt.show()

    @classmethod
    def print_avg_delta(cls):
        val = sum(cls.delta_lens) / len(cls.delta_lens)
        print(val)
        print("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
        return val

    @classmethod
    def tearDownClass(cls) -> None:
        cls.plot_success_failure_bars()
        cls.plot_ious_bar()
        cls.print_avg_delta()

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
                # present_results(detections,png_file)

            # Process XML file
            reference_boxes, test_boxes, test_areas = [], [], []
            if xml_file:
                # list the correct empty parking spots
                reference_boxes = self.parse_bounding_boxes(xml_file)
                #present_results(reference_boxes, png_file)

                # list the empty parking spots from our algorithm
                parking_areas = parking_mark(int(scene_path[-1]), self.parking_area_path)
                test_boxes, test_areas = find_empty_spots(png_file, detections, masks, parking_areas)

            # compare the number of detected empty spots
            len_comparison = len(test_boxes) == len(reference_boxes)
            self.delta_lens.append(abs(len(reference_boxes)-len(test_boxes)))

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
                    self.ious_values.append(iou_value)
                    reference_boxes.remove(reference)
                    self.success_dict[test_areas[i]] += 1
                else:
                    self.ious_values.append(0)
                    self.failure_dict[test_areas[i]] += 1

            # case test_boxes < reference_boxes
            parking_areas_bbox = []
            parking_areas_posture = []
            for parking_area in parking_areas:
                parking_areas_posture.append(parking_area[0])
                parking_areas_bbox.append(parking_area[1])

            if len(reference_boxes) > 0:
                for j in range(len(parking_areas)):
                    failures = detections_in_area(reference_boxes, parking_areas_bbox[j])
                    self.failure_dict[parking_areas_posture[j]] += len(failures)
                    self.ious_values += [0] * len(failures)

            ious.append(len_comparison)
            return ious

    def internal_test_mixed_code(self, test_path):
        if os.path.isdir(test_path):
            detections, annotated_image = None, None
            png_file = None
            masks = []
            xml_file = os.path.join(test_path, "empty_spots.xml")

            # Collect PNG and XML files
            for file in os.listdir(test_path):
                if file.endswith('.png') or file.endswith('.jpg'):
                    png_file = os.path.join(test_path, file)

            # Process PNG files
            if png_file:
                detections, masks, annotated_image = predict(png_file)
                #emptySpots.present_results(detections,png_file)

            # list the correct empty parking spots
            reference_boxes = self.parse_bounding_boxes(xml_file)
            #present_results(reference_boxes,png_file)

            # list the empty parking spots from our algorithm
            parking_areas = mixed_parking_mark(test_path)
            test_boxes, test_areas = find_empty_spots(png_file, detections, masks, parking_areas)

            # compare the number of detected empty spots
            len_comparison = len(test_boxes) == len(reference_boxes)
            self.delta_lens.append(abs(len(reference_boxes)-len(test_boxes)))

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
                    self.ious_values.append(iou_value)
                    reference_boxes.remove(reference)
                    self.success_dict[test_areas[i]] += 1
                else:
                    self.ious_values.append(0)
                    self.failure_dict[test_areas[i]] += 1

            # case test_boxes < reference_boxes
            parking_areas_bbox = []
            parking_areas_posture = []
            for parking_area in parking_areas:
                parking_areas_posture.append(parking_area[0])
                parking_areas_bbox.append(parking_area[1])

            if len(reference_boxes) > 0:
                for j in range(len(parking_areas)):
                    failures = detections_in_area(reference_boxes, parking_areas_bbox[j])
                    self.failure_dict[parking_areas_posture[j]] += len(failures)
                    self.ious_values += [0] * len(failures)

            ious.append(len_comparison)
            return ious

    def internal_test_mixed_code_(self, test_path):
        if os.path.isdir(test_path):
            detections, annotated_image = None, None
            png_file = None
            masks = []
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

            # compare the number of detected empty spots
            len_comparison = len(test_boxes) == len(reference_boxes)
            self.delta_lens.append(len_comparison)

            ious = []
            # key=test_box : value=test_area
            test_boxes_dict = {test_boxes[i]:test_areas[i] for i in range(len(test_boxes))}
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
                    self.ious_values.append(iou_value)
                    self.success_dict[test_areas[i]] += 1
                    reference_boxes.remove(reference)
                    test_boxes_dict.pop(test_boxes[i])

            # here in dictionary only false test_boxes
            test_boxes = [key for key in test_boxes_dict]
            for i in range(len(test_boxes)):
                max_iou_val, idx = 0, 0
                if not reference_boxes:
                    break
                for j in range(len(reference_boxes)):
                    iou_value = self.calculate_iou(test_boxes[i], reference[j])
                    if max_iou_val < iou_value:
                        max_iou_val, idx = iou_value, j
                self.ious_values.append(max_iou_val)
                self.failure_dict[test_boxes_dict[test_boxes[i]]] += 1
                # reference_boxes.remove(reference_boxes[idx])

            # case test_boxes < reference_boxes
            if reference_boxes > 0:
                for area_key in self.success_dict:
                    failures = detections_in_area(reference_boxes, area_key)
                    self.failure_dict[area_key] += len(failures)

            ious.append(len_comparison)
            return ious

