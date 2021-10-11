import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf
from pytorch_lightning import Callback, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.nih_classification_data_module import NIHClassificationDataModule
from modules.clearml_module import ClearMLModule
from modules.nih_efficient_net_v2_module import NIHEfficientNetV2Module
from modules.nih_resnet_module import NIHResNetModule
from train_scripts.common_training_object import CommonTrainingObject
from utils.arg_launcher import ArgLauncher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NIHClassificationTrainingObject(CommonTrainingObject):

    def _setup_data_module(self) -> NIHClassificationDataModule:
        data_module = NIHClassificationDataModule(
            dataset_path=self.cfg.data.dataset_path,
            split_type=self.cfg.data.split_type,
            phases=self.cfg.hparams.phases,
            num_workers=self.cfg.cluster.cpus_per_node,
            classes=self.cfg.data.classes
        )

        self.cfg.hparams.dynamic.classes = data_module.classes
        self.cfg.hparams.dynamic.class_freq = data_module.get_train_class_freq()

        return data_module

    def _setup_model_class(self) -> type(ClearMLModule):
        if self.cfg.hparams.architecture == 'eff_net_v2':
            model_class = NIHEfficientNetV2Module
        elif self.cfg.hparams.architecture == 'resnet':
            model_class = NIHResNetModule
        else:
            raise ValueError()

        return model_class

    def _setup_model_checkpoint(self) -> Optional[ModelCheckpoint]:
        return ModelCheckpoint(
            filename='epoch={epoch}_val_auroc={auroc_avg/val:.3f}_top',
            monitor='auroc_avg/val',
            mode='max',
            dirpath=self.paths.checkpoint_dir,
            verbose=True,
            save_top_k=1,
            auto_insert_metric_name=False
        )

    def _setup_callbacks(self) -> List[Callback]:
        callbacks = super()._setup_callbacks()

        es_cfg = self.cfg.training.early_stopping
        if es_cfg.enabled:
            es_cb = EarlyStopping(
                monitor="auroc_avg/val",
                min_delta=es_cfg.min_delta,
                patience=es_cfg.patience,
                verbose=True,
                mode="max"
            )
            callbacks.append(es_cb)

        return callbacks

    def _load_model_before_test(self) -> LightningModule:
        best_ckpt_path = Path(self.model_checkpoint.best_model_path)
        model = self.model_class.load_from_checkpoint(
            (Path(self.paths.checkpoint_dir) / best_ckpt_path).as_posix())

        return model

    def _upload_output_model(self):
        best_ckpt_path = Path(self.model_checkpoint.best_model_path)
        weights_path = Path(self.paths.checkpoint_dir) / (best_ckpt_path.stem + '.pt')

        if self.trainer.is_global_zero:
            self.model.save_state_to_file(self.model_checkpoint.best_model_path)
            self.task.update_output_model(weights_path.as_posix(), tags=['auroc_best'])


class NIHClassificationTrainingLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument('--name',
                            type=str, default='train',
                            help='Task name.')
        parser.add_argument('--offline', action='store_true')
        parser.add_argument('--on-cluster', action='store_true')

    def run(self, args) -> None:
        cfg_root = Path('config')
        cls_cfg_root = cfg_root / 'classification'

        config = OmegaConf.load(cls_cfg_root / 'train_config.yaml')
        config.cluster = OmegaConf.load(cfg_root / 'train_cluster.yaml')
        config.task_name = args.name

        training_obj = NIHClassificationTrainingObject(
            project_name='Nih-classification',
            cfg=config,
            run_offline=args.offline,
            run_cluster=args.on_cluster
        )
        training_obj.train_and_test()


if __name__ == '__main__':
    NIHClassificationTrainingLauncher(sys.argv[1:]).launch()
