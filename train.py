import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

import clearml
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from callbacks.lr_callbacks import LrDecay, LrWarmup, LrExponential
from data.nih_data_module import NIHDataModule
from loggers.clearml_logger import ClearMLLogger
from models.efficient_net_v2_module import EfficientNetV2Module
from utils.arg_launcher import ArgLauncher
from utils.misc import to_omega_conf
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def connect_with_task(cfg: DictConfig) -> clearml.Task:
    task_args = dict(auto_connect_frameworks=False,
                     output_uri=True)

    if cfg.training.restore_training.enabled:
        task_args.update(dict(continue_last_task=cfg.training.restore_training.task_id))
    else:
        task_args.update(dict(project_name='Nih-classification',
                              task_name=cfg.cluster.job_name))

    clearml.Task.force_requirements_env_freeze(requirements_file='requirements.txt')
    task = clearml.Task.init(**task_args)

    cfg.cluster = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.cluster), name='cluster_cfg'))
    cfg.hparams = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.hparams), name='hparams'))
    cfg.data = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.data), name='data_cfg'))
    cfg.training = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.training), name='training_cfg'))

    task.execute_remotely(exit_process=True)

    return task


def train(cfg: DictConfig):
    pl.seed_everything(42)

    task = connect_with_task(cfg)

    log_root_dir = Path(task.session.config.get('sdk.storage.log_dir')) / task.task_id
    checkpoint_dir = log_root_dir / 'checkpoints'
    log_dir = log_root_dir / 'training_logs'

    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True, parents=True)

    es_cfg = cfg.training.early_stopping

    lr_decay_cfg = cfg.hparams.lr_decay
    lr_exponential_cfg = cfg.hparams.lr_exponential
    lr_warmup_cfg = cfg.hparams.lr_warmup

    checkpoint_file = (checkpoint_dir / cfg.training.restore_training.ckpt_name
                       if cfg.training.restore_training.enabled else None)

    #
    # Define callbacks
    #
    callbacks = []

    max_auc_ckpt_cb = ModelCheckpoint(
        filename='epoch={epoch}_val_auroc={auroc_avg/val:.3f}_top',
        monitor='auroc_avg/val',
        mode='max',
        dirpath=checkpoint_dir,
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

    # Disabled since neptune has its own hardware monitor.
    # gpu_monitor_cb = GPUStatsMonitorEx(prog_bar_filters=['memory.free'])
    # callbacks.append(gpu_monitor_cb)

    # Note: LrDecay callback should be executed before LrWarmup
    lr_decay_cb = LrDecay(
        rate=lr_decay_cfg.rate,
        interval=lr_decay_cfg.interval,
        initial_lr=cfg.hparams.lr_initial
    )
    if lr_decay_cfg.enabled:
        callbacks.append(lr_decay_cb)

    # Note: LrExponential callback should be executed before LrWarmup
    lr_exponential_cb = LrExponential(
        gamma=lr_exponential_cfg.gamma,
        warmup_steps=lr_warmup_cfg.warmup_steps if lr_warmup_cfg.enabled else None,
        phases=cfg.hparams.phases,
        initial_lr=cfg.hparams.lr_initial
    )
    if lr_exponential_cfg.enabled:
        callbacks.append(lr_exponential_cb)

    lr_warmup_cb = LrWarmup(
        warmup_steps=lr_warmup_cfg.warmup_steps,
        phases=cfg.hparams.phases,
        initial_lr=cfg.hparams.lr_initial
    )
    if lr_warmup_cfg.enabled:
        callbacks.append(lr_warmup_cb)

    #
    # Instantiate modules
    #
    dm = NIHDataModule(
        dataset_path=cfg.data.dataset_path,
        split_type=cfg.data.split_type,
        phases=cfg.hparams.phases,
        num_workers=cfg.cluster.cpus_per_node,
        classes=cfg.data.classes
    )

    if cfg.hparams.architecture == 'eff_net_v2':
        cfg.hparams.dynamic.classes = dm.classes
        cfg.hparams.dynamic.class_freq = dm.get_train_class_freq()
        model = EfficientNetV2Module(hparams=cfg.hparams)
    else:
        raise ValueError()

    trainer_params_cluster = dict(
        gpus=cfg.cluster.gpus_per_node,
        num_nodes=cfg.cluster.nodes,
        accelerator='ddp',
        deterministic=True,
        prepare_data_per_node=True,
    )
    trainer_params_common = dict(
        max_epochs=cfg.hparams.epochs,
        logger=ClearMLLogger(task, log_hyperparams=False),
        num_sanity_val_steps=0,
        callbacks=callbacks,
        weights_save_path=checkpoint_dir,
        default_root_dir=log_dir,
        resume_from_checkpoint=checkpoint_file,
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=25
    )
    trainer = Trainer(
        **trainer_params_common,
        **trainer_params_cluster
    )

    trainer.fit(model, datamodule=dm)

    # wait for checkpoint to be saved
    time.sleep(5)

    if cfg.hparams.architecture == 'eff_net_v2':
        best_model_name = Path(max_auc_ckpt_cb.best_model_path)
        best_model_path = checkpoint_dir / (best_model_name.stem + '.pt')
        model = EfficientNetV2Module.load_from_checkpoint((checkpoint_dir / best_model_name).as_posix())
    else:
        raise ValueError()
    if trainer.is_global_zero:
        model.save_to_file(best_model_path)
        task.update_output_model(str(best_model_path), tags=['auroc_best'])

    trainer.test(model, datamodule=dm)

    task.flush()


class NIHTrainingLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument('config_dir', type=str, help='Path to directory with YAML config files.')

    def run(self, args) -> None:
        cfg_root = Path(args.config_dir)

        config = OmegaConf.load(cfg_root / 'train_config.yaml')
        config.cluster = OmegaConf.load(cfg_root / 'cluster.yaml')

        train(config)


if __name__ == '__main__':
    NIHTrainingLauncher(sys.argv[1:]).launch()
