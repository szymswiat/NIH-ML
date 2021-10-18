import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load

from data.nih_art_detection_data_module import NIHArtDetectionDataModule
from training.common_training_module import CommonTrainingModule
from inference.models.faster_rcnn_module import FasterRCNNModule
from training.nih_detection_training_module import NIHDetectionTrainingModule
from training.common_training_object import CommonTrainingObject
from utils.arg_launcher import ArgLauncher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NIHArtDetectionTrainingObject(CommonTrainingObject):

    def __init__(self, project_name: str, cfg: DictConfig, run_offline: bool, run_cluster: bool, run_remote: bool):
        super().__init__(project_name, cfg, run_offline, run_cluster, run_remote)

        self.model_class = FasterRCNNModule

    def _setup_training_module(self) -> CommonTrainingModule:
        model = self.model_class(self.cfg.hparams)
        training_module = NIHDetectionTrainingModule(self.cfg.hparams, model)

        return training_module

    def _load_training_module_before_test(self) -> LightningModule:
        if self.model_checkpoint is None:
            return self.training_module

        best_ckpt_path = Path(self.model_checkpoint.best_model_path)

        checkpoint = pl_load((Path(self.paths.checkpoint_dir) / best_ckpt_path).as_posix())

        training_module = NIHDetectionTrainingModule._load_model_state(
            checkpoint=checkpoint,
            model=self.model_class(self.cfg.hparams)
        )

        return training_module

    def _setup_data_module(self) -> NIHArtDetectionDataModule:
        data_module = NIHArtDetectionDataModule(
            dataset_path=self.cfg.data.dataset_path,
            split_type=self.cfg.data.split_type,
            phases=self.cfg.hparams.phases,
            num_workers=self.cfg.cluster.cpus_per_node
        )

        self.cfg.hparams.dynamic.classes = data_module.classes

        return data_module

    def _setup_model_checkpoint(self) -> Optional[ModelCheckpoint]:
        return ModelCheckpoint(
            filename='epoch={epoch}_val_map={m_ap/all_val:.3f}_top',
            monitor='m_ap/all_val',
            mode='max',
            dirpath=self.paths.checkpoint_dir,
            verbose=True,
            save_top_k=1,
            auto_insert_metric_name=False
        )

    # def _setup_callbacks(self) -> List[Callback]:
    #     callbacks = super()._setup_callbacks()
    #
    #     es_cfg = self.cfg.training.early_stopping
    #     if es_cfg.enabled:
    #         es_cb = EarlyStopping(
    #             monitor="auroc_avg/val",  # TODO:
    #             min_delta=es_cfg.min_delta,
    #             patience=es_cfg.patience,
    #             verbose=True,
    #             mode="max"
    #         )
    #         callbacks.append(es_cb)
    #
    #     return callbacks

    def _upload_output_model(self):
        best_ckpt_path = Path(self.model_checkpoint.best_model_path)
        weights_path = Path(self.paths.checkpoint_dir) / (best_ckpt_path.stem + '.pt')

        if self.trainer.is_global_zero:
            self.training_module.model.save_state_to_file(weights_path)
            self.task.update_output_model(weights_path.as_posix(), tags=['best_map'])


class NIHArtDetectionTrainingLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument('--name',
                            type=str, default='train',
                            help='Task name.')
        parser.add_argument('--offline', action='store_true')
        parser.add_argument('--on-cluster', action='store_true')
        parser.add_argument('--remote', action='store_true')

    def run(self, args) -> None:
        cfg_root = Path('config')
        cls_cfg_root = cfg_root / 'art_detection'

        config = OmegaConf.load(cls_cfg_root / 'train_config.yaml')
        config.cluster = OmegaConf.load(cfg_root / 'train_cluster.yaml')
        config.task_name = args.name

        training_obj = NIHArtDetectionTrainingObject(
            project_name='Nih-art-detection',
            cfg=config,
            run_offline=args.offline,
            run_cluster=args.on_cluster,
            run_remote=args.remote
        )
        training_obj.train_and_test()


if __name__ == '__main__':
    NIHArtDetectionTrainingLauncher(sys.argv[1:]).launch()
