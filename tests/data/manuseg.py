import random

import cv2
import numpy
import scipy.io as sio

from Data import get_monuseg_dicts
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer


def show_image_test():
    TEST_DATASET_ROOT = "/media/serdar/Works2T/Datasets/Nuclei/MoNuSeg2018/Test"
    DatasetCatalog.register("monuseg", lambda dataset_path=TEST_DATASET_ROOT: get_monuseg_dicts(dataset_path))
    MetadataCatalog.get("monuseg").set(thing_classes=["nuclei"])
    balloon_metadata = MetadataCatalog.get("monuseg")

    dataset_dicts = DatasetCatalog.get("monuseg")
    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        inst_map = sio.loadmat(d["file_name"].replace(".tif", ".mat").replace("Images", "Labels"))["inst_map"][..., None].repeat(3, axis=2)

        visualizer = Visualizer(img[..., ::-1], metadata=balloon_metadata, scale=1)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("", out.get_image()[..., ::-1])
        cv2.imshow("inst_map", inst_map.astype(numpy.uint8))

        cv2.waitKey(100)
    return


if __name__ == '__main__':
    show_image_test()
