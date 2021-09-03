import sys
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data.nih_data_module import NIHDataModule
from models.efficient_net_v2_module import EfficientNetV2Module
from utils.arg_launcher import ArgLauncher


def test(cfg: DictConfig):
    pl.seed_everything(42)

    #
    # Extract and setup configuration from config file
    #
    log_ver = cfg.testing.version
    rfc = cfg.testing.ckpt_file
    exp_root_dir = Path(cfg.testing.log_dir) / cfg.testing.job_name

    version = f'version_{log_ver}' if isinstance(log_ver, int) else log_ver

    exp_str = Path(rfc).parts[0]

    log_dir = exp_root_dir / exp_str / version
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_file = exp_root_dir / rfc if rfc else None

    params = cfg.testing.params

    dm = NIHDataModule(
        dataset_path=cfg.data.dataset_path,
        phases=OmegaConf.create([dict(
            image_size=params.image_size,
            augment_rate=0,
            batch_size=params.batch_size,
            epoch_milestone=0
        )]),
        df_prefix=cfg.data.df_prefix
    )

    arch = cfg.testing.architecture

    model_kwargs = dict(class_freq=dm.get_train_class_freq(), num_classes=NIHDataModule.NUM_CLASSES)
    if arch == 'eff_net_v2':
        model = EfficientNetV2Module.load_from_checkpoint(checkpoint_file, **model_kwargs)
    else:
        raise ValueError()

    trainer = Trainer(
        logger=[TensorBoardLogger(save_dir=exp_root_dir,
                                  name=exp_str,
                                  version=log_ver)],
        gpus=1,
        deterministic=True,
        default_root_dir=log_dir,
        weights_save_path=checkpoint_dir
    )

    trainer.test(model, datamodule=dm)


class NIHTestingLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument('config', type=str, help='Path to YAML config file.')

    def run(self, args) -> None:
        config = OmegaConf.load(args.config)

        test(config)


if __name__ == '__main__':
    NIHTestingLauncher(sys.argv[1:]).launch()
