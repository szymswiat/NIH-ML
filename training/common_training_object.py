import os
import time
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from clearml import Task
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning import Callback, LightningDataModule, LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.lr_callbacks import LrDecay, LrWarmup, LrExponential
from loggers.clearml_logger import ClearMLLogger
from training.common_training_module import CommonTrainingModule
from utils.misc import to_omega_conf


class CommonTrainingObject:

    def __init__(
            self,
            project_name: str,
            cfg: DictConfig,
            run_offline: bool,
            run_cluster: bool,
            run_remote: bool
    ):
        self.project_name = project_name
        self.cfg = cfg
        self.run_offline = run_offline
        self.run_cluster = run_cluster
        self.run_remote = run_remote

        self.task: Task = None
        self.paths: DictConfig = None
        self.callbacks: List[Callback] = None
        self.model_checkpoint: ModelCheckpoint = None
        self.data_module: LightningDataModule = None
        self.training_module: CommonTrainingModule = None
        self.trainer: Trainer = None

    def setup_training_objects(self):
        self.task = self._connect_with_task()
        self.paths = self._setup_paths()
        self.callbacks = self._setup_callbacks()

        self.model_checkpoint = self._setup_model_checkpoint()
        if self.model_checkpoint is not None:
            self.callbacks.append(self.model_checkpoint)

        self.data_module = self._setup_data_module()

        self.training_module = self._setup_training_module()

        self.trainer = self._setup_trainer(self.callbacks)

    @abstractmethod
    def _setup_data_module(self) -> LightningDataModule:
        raise NotImplementedError()

    @abstractmethod
    def _setup_training_module(self) -> CommonTrainingModule:
        raise NotImplementedError()

    @abstractmethod
    def _setup_model_checkpoint(self) -> Optional[ModelCheckpoint]:
        raise NotImplementedError()

    def _load_training_module_before_test(self) -> LightningModule:
        return self.training_module

    @abstractmethod
    def _upload_output_model(self):
        raise NotImplementedError()

    def train_and_test(self):
        pl.seed_everything(42)

        self.setup_training_objects()

        self.trainer.fit(self.training_module, datamodule=self.data_module)

        # wait for checkpoint to be saved (required when using ddp)
        time.sleep(5)

        self.training_module = self._load_training_module_before_test()
        self._upload_output_model()

        self.trainer.test(self.training_module, datamodule=self.data_module)

        self.task.flush()

    def _connect_with_task(self) -> Task:
        task_args = dict(auto_connect_frameworks=False,
                         output_uri=True)

        if self.cfg.training.restore_training.enabled:
            task_args.update(dict(continue_last_task=self.cfg.training.restore_training.task_id))
        else:
            task_args.update(dict(project_name=self.project_name,
                                  task_name=self.cfg.task_name))

        if self.run_offline:
            Task.set_offline(True)

        Task.force_requirements_env_freeze(requirements_file='requirements.txt')
        task = Task.init(**task_args)

        self.cfg.cluster = to_omega_conf(task.connect_configuration(
            OmegaConf.to_object(self.cfg.cluster), name='cluster_cfg'))
        self.cfg.hparams = to_omega_conf(task.connect_configuration(
            OmegaConf.to_object(self.cfg.hparams), name='hparams'))
        self.cfg.training = to_omega_conf(task.connect_configuration(
            OmegaConf.to_object(self.cfg.training), name='training_cfg'))

        if self.run_remote:
            task.execute_remotely(exit_process=True)

        return task

    def _setup_paths(self) -> DictConfig:
        paths = DictConfig({})

        log_root_dir = Path(os.path.expandvars(self.task.session.config.get('sdk.storage.log_dir'))
                            ).expanduser() / self.task.task_id
        checkpoint_dir = log_root_dir / 'checkpoints'
        log_dir = log_root_dir / 'training_logs'

        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        log_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_file = (checkpoint_dir / self.cfg.training.restore_training.ckpt_name
                           if self.cfg.training.restore_training.enabled else None)

        paths.log_root_dir = log_root_dir.as_posix()
        paths.checkpoint_dir = checkpoint_dir.as_posix()
        paths.log_dir = log_dir.as_posix()
        paths.checkpoint_file = checkpoint_file.as_posix() if checkpoint_file else None

        return paths

    def _setup_trainer(self, callbacks: List[Callback]) -> Trainer:
        trainer_params = dict(
            max_epochs=self.cfg.hparams.epochs,
            logger=ClearMLLogger(self.task, log_hyperparams=False),
            num_sanity_val_steps=0,
            callbacks=callbacks,
            weights_save_path=self.paths.checkpoint_dir,
            default_root_dir=self.paths.log_dir,
            resume_from_checkpoint=self.paths.checkpoint_file,
            reload_dataloaders_every_n_epochs=1,
            log_every_n_steps=25,
            # limit_train_batches=10
        )
        if self.run_cluster:
            trainer_params.update(dict(
                gpus=self.cfg.cluster.gpus_per_node,
                num_nodes=self.cfg.cluster.nodes,
                accelerator='ddp',
                deterministic=True,
                prepare_data_per_node=True,
            ))
        return Trainer(**trainer_params)

    def _setup_callbacks(self) -> List[Callback]:
        lr_decay_cfg = self.cfg.hparams.optimizer.lr_decay
        lr_exponential_cfg = self.cfg.hparams.optimizer.lr_exponential
        lr_warmup_cfg = self.cfg.hparams.optimizer.lr_warmup

        callbacks = []

        # Note: LrDecay callback should be executed before LrWarmup
        lr_decay_cb = LrDecay(
            rate=lr_decay_cfg.rate,
            interval=lr_decay_cfg.interval,
            initial_lr=self.cfg.hparams.optimizer.lr_initial
        )
        if lr_decay_cfg.enabled:
            callbacks.append(lr_decay_cb)

        # Note: LrExponential callback should be executed before LrWarmup
        lr_exponential_cb = LrExponential(
            gamma=lr_exponential_cfg.gamma,
            warmup_steps=lr_warmup_cfg.warmup_steps if lr_warmup_cfg.enabled else None,
            phases=self.cfg.hparams.phases,
            initial_lr=self.cfg.hparams.optimizer.lr_initial
        )
        if lr_exponential_cfg.enabled:
            callbacks.append(lr_exponential_cb)

        lr_warmup_cb = LrWarmup(
            warmup_steps=lr_warmup_cfg.warmup_steps,
            phases=self.cfg.hparams.phases,
            initial_lr=self.cfg.hparams.optimizer.lr_initial
        )
        if lr_warmup_cfg.enabled:
            callbacks.append(lr_warmup_cb)

        return callbacks
