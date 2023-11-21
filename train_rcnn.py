import logging
import math
import operator
import os
from collections import OrderedDict

from fvcore.common.checkpoint import Checkpointer

import detectron2.utils.comm as comm
from Data import get_midog_yolo_dicts, get_monuseg_dicts
from detectron2.checkpoint import DetectionCheckpointer
from configs import get_cfg
# from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, HookBase
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


class BestCheckpointer(HookBase):
    """
    Checkpoints best weights based off given metric.
    This hook should be used in conjunction to and executed after the hook
    that produces the metric, e.g. `EvalHook`.
    """

    def __init__(
            self,
            eval_period: int,
            checkpointer: Checkpointer,
            val_metric: str,
            mode: str = "max",
            file_prefix: str = "model_best",
    ) -> None:
        """
        Args:
            eval_period (int): the period `EvalHook` is set to run.
            checkpointer: the checkpointer object used to save checkpoints.
            val_metric (str): validation metric to track for best checkpoint, e.g. "bbox/AP50"
            mode (str): one of {'max', 'min'}. controls whether the chosen val metric should be
                maximized or minimized, e.g. for "bbox/AP50" it should be "max"
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
        """
        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metric = val_metric
        assert mode in [
            "max",
            "min",
        ], f'Mode "{mode}" to `BestCheckpointer` is unknown. It should be one of {"max", "min"}.'
        if mode == "max":
            self._compare = operator.gt
        else:
            self._compare = operator.lt
        self._checkpointer = checkpointer
        self._file_prefix = file_prefix
        self.best_metric = None
        self.best_iter = None

    def _update_best(self, val, iteration):
        if math.isnan(val) or math.isinf(val):
            return False
        self.best_metric = val
        self.best_iter = iteration
        return True

    def _best_checking(self):
        metric_tuple = self.trainer.storage.latest().get(self._val_metric)
        if metric_tuple is None:
            self._logger.warning(
                f"Given val metric {self._val_metric} does not seem to be computed/stored."
                "Will not be checkpointing based on it."
            )
            return
        else:
            latest_metric, metric_iter = metric_tuple

        if self.best_metric is None:
            if self._update_best(latest_metric, metric_iter):
                additional_state = {"iteration": metric_iter}
                self._checkpointer.save(f"{self._file_prefix}", **additional_state)
                self._logger.info(
                    f"Saved first model at {self.best_metric:0.5f} @ {self.best_iter} steps"
                )
        elif self._compare(latest_metric, self.best_metric):
            additional_state = {"iteration": metric_iter}
            self._checkpointer.save(f"{self._file_prefix}", **additional_state)
            self._logger.info(
                f"Saved best model as latest eval score for {self._val_metric} is"
                f"{latest_metric:0.5f}, better than last best score "
                f"{self.best_metric:0.5f} @ iteration {self.best_iter}."
            )
            self._update_best(latest_metric, metric_iter)
        else:
            self._logger.info(
                f"Not saving as latest eval score for {self._val_metric} is {latest_metric:0.5f}, "
                f"not better than best score {self.best_metric:0.5f} @ iteration {self.best_iter}."
            )

    def after_step(self):
        # same conditions as `EvalHook`
        next_iter = self.trainer.iter + 1
        if (
                self._period > 0
                and next_iter % self._period == 0
                and next_iter != self.trainer.max_iter
        ):
            self._best_checking()

    def after_train(self):
        # same conditions as `EvalHook`
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._best_checking()


def register_datasets(cfg):
    if cfg.DATASETS.MIDOG_TRAIN_DATASET_ROOT is not None:
        TRAIN_DATASET_ROOT = cfg.DATASETS.MIDOG_TRAIN_DATASET_ROOT
        DatasetCatalog.register("midog_yolo_train",
                                lambda dataset_path=TRAIN_DATASET_ROOT: get_midog_yolo_dicts(dataset_path))
        MetadataCatalog.get("midog_yolo_train").set(thing_classes=["mitosis", "hard-negative"])
        MetadataCatalog.get("midog_yolo_train").set(evaluator_type="coco")

    if cfg.DATASETS.MIDOG_TEST_DATASET_ROOT is not None:
        TEST_DATASET_ROOT = cfg.DATASETS.MIDOG_TEST_DATASET_ROOT
        DatasetCatalog.register("midog_yolo_val", lambda dataset_path=TEST_DATASET_ROOT: get_midog_yolo_dicts(dataset_path))
        MetadataCatalog.get("midog_yolo_val").set(thing_classes=["mitosis", "hard-negative"])
        MetadataCatalog.get("midog_yolo_val").set(evaluator_type="coco")

    if cfg.DATASETS.MONUSEG_TRAIN_DATASET_ROOT is not None:
        TRAIN_DATASET_ROOT = cfg.DATASETS.MONUSEG_TRAIN_DATASET_ROOT
        DatasetCatalog.register("monuseg_train", lambda dataset_path=TRAIN_DATASET_ROOT: get_monuseg_dicts(dataset_path))
        MetadataCatalog.get("monuseg_train").set(thing_classes=["nuclei"])
        MetadataCatalog.get("monuseg_train").set(evaluator_type="coco")

    if cfg.DATASETS.MONUSEG_TEST_DATASET_ROOT is not None:
        TEST_DATASET_ROOT = cfg.DATASETS.MONUSEG_TEST_DATASET_ROOT
        DatasetCatalog.register("monuseg_val", lambda dataset_path=TEST_DATASET_ROOT: get_monuseg_dicts(dataset_path))
        MetadataCatalog.get("monuseg_val").set(thing_classes=["nuclei"])
        MetadataCatalog.get("monuseg_val").set(evaluator_type="coco")

    return


def main(args):
    cfg = setup(args)
    register_datasets(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.register_hooks(
        [hooks.EvalHook(cfg.TEST.EVAL_PERIOD, lambda: trainer.test_with_TTA(cfg, trainer.model)),
         BestCheckpointer(cfg.TEST.EVAL_PERIOD, trainer.checkpointer, "bbox/AP50", mode="max")]
    )

    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(cfg.TEST.EVAL_PERIOD, lambda: trainer.test_with_TTA(cfg, trainer.model)),
             BestCheckpointer(cfg.TEST.EVAL_PERIOD, trainer.checkpointer, "bbox/AP50", mode="max")]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
