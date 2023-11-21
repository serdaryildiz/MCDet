import os
from typing import List

import cv2
import numpy
import tqdm
import scipy.io as sio

from detectron2.structures import BoxMode


def get_monuseg_dicts(dataset_root: str) -> List[dict]:
    """
        return MaNuSeg dataset
    :param dataset_root: dataset path
    :return: dataset dict
    """

    images_root = os.path.join(dataset_root, "Images")  # .tif files
    image_names = os.listdir(images_root)
    labels_root = os.path.join(dataset_root, "Labels")  # .mat files

    return_list = []
    for image_id, image_name in enumerate(image_names):
        image_path = os.path.join(images_root, image_name)
        label_path = os.path.join(labels_root, image_name.replace(".tif", ".mat"))

        # get image width and height
        height, width = cv2.imread(image_path).shape[:2]

        # read label matrix
        inst_map = sio.loadmat(label_path)["inst_map"]
        max_id = numpy.max(inst_map)

        objects = []
        for id in range(1, max_id + 1):
            # get object mask
            mask = numpy.where(inst_map == id, 1, 0).astype(numpy.uint8)

            # find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # get coordinates
            px = []
            py = []
            for points in contours:
                for point in points:
                    px.append(point[0][0])
                    py.append(point[0][1])
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [float(p) for x in poly for p in x]

            # get bounding box
            bbox = [float(numpy.min(px)),  # x1
                    float(numpy.min(py)),  # y1
                    float(numpy.max(px)),  # x2
                    float(numpy.max(py))   # y2
                    ]

            # just nuclei
            category_id = 0

            # some detectron2 problems...
            if len(poly) >= 6:
                objects.append(
                    {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "segmentation": [poly],
                        "category_id": category_id,
                    }
                )

        # add sample
        return_list.append({
            "file_name": image_path,
            "image_id": image_id,
            "height": height,
            "width": width,
            "annotations": objects
        })

    return return_list









