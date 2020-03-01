import xml.etree.ElementTree as ET
import numpy as np
import glob
import os


def parse_xml(annot_dir):
    """
    Parse XML annotation files in VOC dataset

    :param annot_dir: directory path to annotation files
    :return: 2-d array
    """
    boxes = []

    for xml_file in glob.glob(os.path.join(annot_dir, '*.xml')):
        tree = ET.parse(xml_file)

        h_img = int(tree.findtext('./size/height'))
        w_img = int(tree.findtext('./size/width'))

        for obj in tree.iter('object'):
            xmin = int(round(float(obj.findtext('bndbox/xmin'))))
            ymin = int(round(float(obj.findtext('bndbox/ymin'))))
            xmax = int(round(float(obj.findtext('bndbox/xmax'))))
            ymax = int(round(float(obj.findtext('bndbox/ymax'))))

            w_norm = (xmax - xmin) / w_img
            h_norm = (ymax - ymin) / h_img

            boxes.append([w_norm, h_norm])

    return np.array(boxes)
