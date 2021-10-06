from typing import List, Dict, Tuple

from omegaconf import DictConfig
from pytorch_lightning import Callback

from callbacks.lr_callbacks import LrDecay, LrWarmup, LrExponential


def setup_common_callbacks(cfg: DictConfig, paths: DictConfig) -> Tuple[List[Callback], Dict]:
    lr_decay_cfg = cfg.hparams.optimizer.lr_decay
    lr_exponential_cfg = cfg.hparams.optimizer.lr_exponential
    lr_warmup_cfg = cfg.hparams.optimizer.lr_warmup

    callbacks = []
    callback_dct = {}

    # Note: LrDecay callback should be executed before LrWarmup
    lr_decay_cb = LrDecay(
        rate=lr_decay_cfg.rate,
        interval=lr_decay_cfg.interval,
        initial_lr=cfg.hparams.optimizer.lr_initial
    )
    if lr_decay_cfg.enabled:
        callbacks.append(lr_decay_cb)

    # Note: LrExponential callback should be executed before LrWarmup
    lr_exponential_cb = LrExponential(
        gamma=lr_exponential_cfg.gamma,
        warmup_steps=lr_warmup_cfg.warmup_steps if lr_warmup_cfg.enabled else None,
        phases=cfg.hparams.phases,
        initial_lr=cfg.hparams.optimizer.lr_initial
    )
    if lr_exponential_cfg.enabled:
        callbacks.append(lr_exponential_cb)

    lr_warmup_cb = LrWarmup(
        warmup_steps=lr_warmup_cfg.warmup_steps,
        phases=cfg.hparams.phases,
        initial_lr=cfg.hparams.optimizer.lr_initial
    )
    if lr_warmup_cfg.enabled:
        callbacks.append(lr_warmup_cb)

    return callbacks, callback_dct
