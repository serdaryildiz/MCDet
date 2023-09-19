import os
from typing import List

import cv2
import numpy
import tqdm

from detectron2.structures import BoxMode


def get_midog_yolo_dicts(dataset_root: str) -> List[dict]:
    """
        return midog dataset
    :param dataset_root:
    :return:
    """
    # root paths
    images_root = os.path.join(dataset_root, "images")
    labels_root = os.path.join(dataset_root, "labels")

    assert os.path.exists(images_root), f"images root {images_root} does not exists!"
    assert os.path.exists(labels_root), f"labels root {labels_root} does not exists!"

    cache_path = os.path.join(dataset_root, "cache_midog.npy")
    if os.path.exists(cache_path):
        return numpy.load(cache_path)

    # label paths
    label_file_names = os.listdir(labels_root)

    return_list = []
    for image_id, label_file_name in tqdm.tqdm(enumerate(label_file_names), total=len(label_file_names)):
        assert label_file_name[-4:] == ".txt"
        label_path = os.path.join(labels_root, label_file_name)

        # read image and get H and W
        image_file_name = label_file_name.replace(".txt", ".tiff")
        image_path = os.path.join(images_root, image_file_name)
        height, width = cv2.imread(image_path).shape[:2]

        # read label
        with open(label_path, "r") as fp:
            lines = fp.readlines()
        fp.close()

        objects = []
        for l in lines:
            obj_class, center_x, center_y, bbox_w, bbox_h = [float(x) for x in l.strip().split(" ")]
            obj_class = int(obj_class)

            # yolo format to x1y1x2y2
            center_x *= width
            center_y *= height
            bbox_w *= width
            bbox_h *= height

            x1 = int(center_x - (bbox_w / 2))
            y1 = int(center_y - (bbox_h / 2))
            x2 = int(x1 + bbox_w)
            y2 = int(y1 + bbox_h)

            # add object
            objects.append({
                "bbox": [x1, y1, x2, y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": obj_class,
            })

        # add sample
        return_list.append({
            "file_name": image_path,
            "image_id": image_id,
            "height": height,
            "width": width,
            "annotations": objects
        })

    # save cache
    numpy.save(cache_path, return_list)

    return return_list





















