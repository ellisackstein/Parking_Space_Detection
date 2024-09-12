from CancelMovingCars import moving_vs_stat
import supervision as sv

import unittest
import os
import xml.etree.ElementTree as ET
from Yolo import predict
from MarkParkingArea import parking_mark
from ExtractEmptySpots import find_empty_spots


class Tests(unittest.TestCase):
    FILE_NAME = ""
    base_dir = 'MovingCars'

    # scene 1
    def test_scene1_test1(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test12")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test2(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test1")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test3(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test2")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test4(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test3")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test5(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test5")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test6(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test6")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test7(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test7")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test8(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test8")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test9(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test9")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test_10(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test6")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test11(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test11")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test12(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test12")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test13(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test13")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test14(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test14")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test15(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test15")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test16(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test16")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test17(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test17")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test18(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test18")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test19(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test19")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test20(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test20")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test21(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test21")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test22(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test22")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test23(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test23")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test24(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test24")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test25(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test25")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test26(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test26")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test27(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test27")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test28(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test28")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test29(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test29")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene1_test30(self):
        scene_path = os.path.join(self.base_dir, "scene1")
        test_path = os.path.join(scene_path, "test30")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test1(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test12")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    # scene 2
    def test_scene2_test2(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test1")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test3(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test2")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test4(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test3")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test5(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test5")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test6(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test6")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test7(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test7")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test8(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test8")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test9(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test9")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test10(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test10")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test11(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test11")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test12(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test12")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test13(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test13")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test14(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test14")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test15(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test15")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test16(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test16")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test17(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test17")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test18(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test18")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test19(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test19")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test20(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test20")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test21(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test21")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test22(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test22")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test23(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test23")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test24(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test24")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test25(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test25")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test26(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test26")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test27(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test27")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test28(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test28")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test29(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test29")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene2_test30(self):
        scene_path = os.path.join(self.base_dir, "scene2")
        test_path = os.path.join(scene_path, "test30")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    # scene 4
    def test_scene4_test2(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test1")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test3(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test2")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test4(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test3")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test5(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test5")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test6(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test6")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test7(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test7")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test8(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test8")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test9(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test9")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test10(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test10")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test11(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test11")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test12(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test12")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test13(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test13")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test14(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test14")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test15(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test15")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test16(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test16")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test17(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test17")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test18(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test18")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test19(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test19")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test20(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test20")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test21(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test21")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test22(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test22")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test23(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test23")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test24(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test24")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test25(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test25")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test26(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test26")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test27(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test27")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test28(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test28")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test29(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test29")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def test_scene4_test30(self):
        scene_path = os.path.join(self.base_dir, "scene4")
        test_path = os.path.join(scene_path, "test30")
        iou_value = self.internal_test_code(scene_path, test_path)
        self.assertTrue(iou_value)

    def parse_bounding_boxes(self, xml_file):
        """
        Parse bounding boxes from an XML annotation file.
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

    def calculate_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Args: box1, box2: Bounding boxes in the format [x1, y1, x2, y2]
        Returns: iou: IoU value
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

    def internal_test_code(self, scene_path, test_path):
        """
        Process each test directory and compare bounding boxes using IoU.
        Args: base_dir (str): Path to the base directory containing test directories.
        """
        if os.path.isdir(test_path):
            detections, masks, annotated_images = [None, None], [None, None], [None, None]
            png_files, xml_file = [], None

            # Collect PNG and XML files
            for file in os.listdir(test_path):
                if file.endswith('.png'):
                    png_files.append(os.path.join(test_path, file))
                elif file.endswith('.xml'):
                    xml_file = os.path.join(test_path, file)

            # Process PNG files
            for i, png_file in enumerate(sorted(png_files)):
                detections[i], masks[i], annotated_images[i] = predict(png_file)

            # Process XML file
            iou_value = 0
            if xml_file:
                reference_box = self.cancel_moving_cars(detections[0], annotated_images[0], detections[1], annotated_images[1])
                test_box = self.parse_bounding_boxes(xml_file)
                if len(reference_box) == 0:
                    return False
                iou_value = self.calculate_iou(test_box, reference_box[0])
            return (iou_value >= 0.7)


    def cancel_moving_cars(self, detections1, annotated_image1, detections2, annotated_image2):
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

