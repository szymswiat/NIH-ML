import logging
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict, Tuple

import clearml
import pytorch_lightning as pl
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data.nih_classification_data_module import NIHClassificationDataModule
from loggers.clearml_logger import ClearMLLogger
from modules.clearml_module import ClearMLModule
from modules.nih_efficient_net_v2_module import NIHEfficientNetV2Module
from modules.nih_resnet_module import NIHResNetModule
from train_scripts.task_connector import connect_with_task
from train_scripts.common_callbacks import setup_common_callbacks
from utils.arg_launcher import ArgLauncher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RUN_OFFLINE = True
RUN_CLUSTER = not RUN_OFFLINE


def train(cfg: DictConfig):
    task = connect_with_task(cfg, project_name='Nih-classification', run_offline=RUN_OFFLINE)

    paths = DictConfig({})

    log_root_dir = Path(os.path.expandvars(task.session.config.get('sdk.storage.log_dir'))).expanduser() / task.task_id
    checkpoint_dir = log_root_dir / 'checkpoints'
    log_dir = log_root_dir / 'training_logs'

    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_file = (checkpoint_dir / cfg.training.restore_training.ckpt_name
                       if cfg.training.restore_training.enabled else None)

    paths.log_root_dir = log_root_dir.as_posix()
    paths.checkpoint_dir = checkpoint_dir.as_posix()
    paths.log_dir = log_dir.as_posix()
    paths.checkpoint_file = checkpoint_file.as_posix() if checkpoint_file else None

    pl.seed_everything(42)

    callbacks, callback_dct = setup_callbacks(cfg, paths)
    data_module = setup_data_module(cfg)
    model_class = setup_model_class(cfg)

    model = model_class(hparams=cfg.hparams)

    trainer = setup_trainer(cfg, task, paths, callbacks)

    trainer.fit(model, datamodule=data_module)

    # wait for checkpoint to be saved
    time.sleep(5)

    best_model_name = Path(callback_dct['model_ckpt'].best_model_path)
    best_model_path = checkpoint_dir / (best_model_name.stem + '.pt')

    model = model_class.load_from_checkpoint((checkpoint_dir / best_model_name).as_posix())

    if trainer.is_global_zero:
        model.save_to_file(best_model_path)
        task.update_output_model(str(best_model_path), tags=['auroc_best'])

    trainer.test(model, datamodule=data_module)

    task.flush()


def setup_data_module(cfg: DictConfig) -> NIHClassificationDataModule:
    data_module = NIHClassificationDataModule(
        dataset_path=cfg.data.dataset_path,
        split_type=cfg.data.split_type,
        phases=cfg.hparams.phases,
        num_workers=cfg.cluster.cpus_per_node,
        classes=cfg.data.classes
    )

    cfg.hparams.dynamic.classes = data_module.classes
    cfg.hparams.dynamic.class_freq = data_module.get_train_class_freq()

    return data_module


def setup_model_class(cfg: DictConfig) -> type(ClearMLModule):
    if cfg.hparams.architecture == 'eff_net_v2':
        model_class = NIHEfficientNetV2Module
    elif cfg.hparams.architecture == 'resnet':
        model_class = NIHResNetModule
    else:
        raise ValueError()

    return model_class


def setup_trainer(
        cfg: DictConfig,
        task: clearml.Task,
        paths: DictConfig,
        callbacks: List[Callback]
) -> Trainer:
    trainer_params = dict(
        max_epochs=cfg.hparams.epochs,
        logger=ClearMLLogger(task, log_hyperparams=False),
        num_sanity_val_steps=0,
        callbacks=callbacks,
        weights_save_path=paths.checkpoint_dir,
        default_root_dir=paths.log_dir,
        resume_from_checkpoint=paths.checkpoint_file,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=25
    )
    if RUN_CLUSTER:
        trainer_params.update(dict(
            gpus=cfg.cluster.gpus_per_node,
            num_nodes=cfg.cluster.nodes,
            accelerator='ddp',
            deterministic=True,
            prepare_data_per_node=True,
        ))
    return Trainer(**trainer_params)


def setup_callbacks(cfg: DictConfig, paths: DictConfig) -> Tuple[List[Callback], Dict]:
    es_cfg = cfg.training.early_stopping

    callbacks, callback_dct = setup_common_callbacks(cfg, paths)

    max_auc_ckpt_cb = ModelCheckpoint(
        filename='epoch={epoch}_val_auroc={auroc_avg/val:.3f}_top',
        monitor='auroc_avg/val',
        mode='max',
        dirpath=paths.checkpoint_dir,
        verbose=True,
        save_top_k=1,
        auto_insert_metric_name=False
    )
    callbacks.append(max_auc_ckpt_cb)

    es_cb = EarlyStopping(
        monitor="auroc_avg/val",
        min_delta=es_cfg.min_delta,
        patience=es_cfg.patience,
        verbose=True,
        mode="max"
    )
    if es_cfg.enabled:
        callbacks.append(es_cb)

    callback_dct['model_ckpt'] = max_auc_ckpt_cb
    return callbacks, callback_dct


class NIHClassificationTrainingLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument('--name',
                            type=str, default='train',
                            help='Task name.')

    def run(self, args) -> None:
        cfg_root = Path('config')
        cls_cfg_root = cfg_root / 'classification'

        config = OmegaConf.load(cls_cfg_root / 'train_config.yaml')
        config.cluster = OmegaConf.load(cfg_root / 'train_cluster.yaml')
        config.task_name = args.name

        train(config)


if __name__ == '__main__':
    NIHClassificationTrainingLauncher(sys.argv[1:]).launch()
