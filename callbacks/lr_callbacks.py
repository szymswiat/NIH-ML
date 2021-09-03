from typing import List, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning import Callback
from torch.optim import Optimizer


class LrInitializer(Callback):

    def __init__(
            self,
            initial_lr: Optional[float] = None
    ):
        self.initial_lr = initial_lr

    def on_train_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        if self.initial_lr is None:
            return

        for opt in trainer.optimizers:
            for group in opt.param_groups:
                group['lr'] = self.initial_lr


class LrDecay(LrInitializer):

    def __init__(
            self,
            rate: float,
            interval: int,
            initial_lr: Optional[float] = None
    ) -> None:
        super().__init__(initial_lr)

        self.rate = rate
        self.interval = interval

    def on_before_optimizer_step(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            optimizer: Optimizer,
            opt_idx: int
    ) -> None:
        batch_idx = trainer._fit_loop.batch_idx
        epoch = trainer.current_epoch

        # decay lr with specified interval
        if batch_idx == 0 and epoch != 0 and epoch % self.interval == 0:
            for group in optimizer.param_groups:
                group['lr'] *= self.rate


class LrExponential(LrInitializer):

    def __init__(
            self,
            gamma: float,
            warmup_steps: Optional[int] = None,
            phases: Optional[Dict] = [],
            initial_lr: Optional[float] = None
    ):
        super().__init__(initial_lr)

        self.initial_lr = initial_lr
        self.gamma = gamma

        self._warmup_steps = warmup_steps
        self._phases = phases

        self._steps_per_train_epoch = -1
        self._warmup_start_step = -1

    def on_train_epoch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:

        for phase in reversed(self._phases):
            if trainer.current_epoch == phase['epoch_milestone']:
                self._warmup_start_step = trainer.global_step
                break

        #
        # Find amount of steps per epoch
        #
        dataloader = trainer.lightning_module.train_dataloader()
        if trainer.max_steps:
            return trainer.max_steps

        if isinstance(trainer.limit_train_batches, float):
            dataset_batches = int(len(dataloader) * trainer.limit_train_batches)
        elif isinstance(trainer.limit_train_batches, int):
            dataset_batches = trainer.limit_train_batches
        else:
            dataset_batches = len(dataloader)

        num_devices = max(1, trainer.num_gpus, trainer.num_processes)
        if trainer.tpu_cores:
            num_devices = max(num_devices, trainer.tpu_cores)

        step_divisor = trainer.accumulate_grad_batches * num_devices

        self._steps_per_train_epoch = (dataset_batches // step_divisor)

    def on_before_optimizer_step(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            optimizer: Optimizer,
            opt_idx: int
    ) -> None:
        global_step = trainer.global_step

        if self._warmup_steps is not None:
            warmup_step = global_step - self._warmup_start_step
            if warmup_step == 0:
                gamma_pow = trainer.current_epoch + self._warmup_steps / self._steps_per_train_epoch
                for group in optimizer.param_groups:
                    group['lr'] = self.initial_lr * self.gamma ** gamma_pow

            if warmup_step < self._warmup_steps:
                return

        batch_idx = trainer._fit_loop.batch_idx
        gamma_pow = trainer.current_epoch + batch_idx / self._steps_per_train_epoch
        for group in optimizer.param_groups:
            group['lr'] = self.initial_lr * self.gamma ** gamma_pow


class LrWarmup(LrInitializer):

    def __init__(
            self,
            warmup_steps: int,
            phases: Optional[Dict] = None,
            initial_lr: Optional[float] = None
    ):
        super().__init__(initial_lr)
        self.warmup_steps = warmup_steps
        self.phases = phases

        self._warmup_start_step = -1
        self._warmup_target: List[float] = []

    def on_train_epoch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        for phase in reversed(self.phases):
            if trainer.current_epoch == phase['epoch_milestone']:
                self._warmup_start_step = trainer.global_step
                break

    def on_before_optimizer_step(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            optimizer: Optimizer,
            opt_idx: int
    ) -> None:

        global_step = trainer.global_step
        warmup_step = global_step - self._warmup_start_step

        # save warmup target at the beginning of each phase
        if warmup_step == 0:
            self._warmup_target = [group['lr'] for group in optimizer.param_groups]

        # perform lr warmup updates until 'lr_warmup_steps' at the beginning of each phase
        if warmup_step <= self.warmup_steps:
            for group, target_lr in zip(optimizer.param_groups, self._warmup_target):
                group['lr'] = warmup_step / self.warmup_steps * target_lr
