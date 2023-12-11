import argparse
import glob
import os

import torch
from tqdm import tqdm

from configs import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine import default_argument_parser, default_setup, DefaultPredictor
from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels

classnames = ["mitosis", "hard-negative"]


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/FasterRCNN/faster_rcnn_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="./input_images/",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="./demo_out/",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.01,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        # self.predictor = TTADefaultPredictor(cfg)
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)

        colors = colormap(rgb=True, maximum=1)
        instances = predictions["instances"].to(self.cpu_device)

        assigned_colors = [colors[i] for i in instances.pred_classes]
        labels = _create_text_labels(
            instances.pred_classes, instances.scores, classnames
        )

        vis_output = visualizer.overlay_instances(
            labels=labels,
            boxes=instances.get("pred_boxes"),
            masks=instances.get("pred_masks"),
            assigned_colors=assigned_colors,
            alpha=0.1
        )
        return predictions, vis_output


def demo(args):
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    args.input = glob.glob(args.input + "/*")
    for path in tqdm(args.input, disable=not args.output):
        img = read_image(path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)

        out_filename = os.path.join(args.output, os.path.basename(path))
        visualized_output.save(out_filename)

    return


if __name__ == '__main__':
    args = get_parser().parse_args()
    demo(args)
