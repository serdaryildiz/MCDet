"""
Source : https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
"""

import random

import cv2

from Data import get_midog_yolo_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def show_image_test():
    TRAIN_DATASET_ROOT = "/media/serdar/BackUp/Datasets/MIDOG2022_Dataset/Samples/val"
    DatasetCatalog.register("midog_subset", lambda dataset_path=TRAIN_DATASET_ROOT: get_midog_yolo_dicts(dataset_path))
    MetadataCatalog.get("midog_subset").set(thing_classes=["mitosis", "hard-negative"])
    balloon_metadata = MetadataCatalog.get("midog_subset")

    dataset_dicts = DatasetCatalog.get("midog_subset")
    for d in random.sample(dataset_dicts, 5):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[..., ::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("", out.get_image()[..., ::-1])
        cv2.waitKey(0)
    return


if __name__ == '__main__':
    show_image_test()
