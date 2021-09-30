import sys
from argparse import ArgumentParser
from pathlib import Path

import clearml
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer

from data.nih_data_module import NIHDataModule
from loggers.clearml_logger import ClearMLLogger
from models.efficient_net_v2_module import EfficientNetV2Module
from utils.arg_launcher import ArgLauncher
from utils.misc import to_omega_conf


def test(test_cfg: DictConfig):
    pl.seed_everything(42)

    train_task = clearml.Task.get_task(task_id=test_cfg.task_id)

    task = clearml.Task.init(project_name=train_task.project,
                             task_name=f'{train_task.name}_test',
                             auto_connect_frameworks=False,
                             output_uri=True)
    task.execute_remotely(exit_process=True)

    hparams = to_omega_conf(train_task._get_configuration_dict('hparams'))
    cluster_cfg = to_omega_conf(train_task._get_configuration_dict('cluster_cfg'))
    data_cfg = to_omega_conf(train_task._get_configuration_dict('data_cfg'))

    log_root_dir = Path(cluster_cfg.log_dir) / task.task_id
    checkpoint_dir = log_root_dir / 'checkpoints'
    log_dir = log_root_dir / 'training_logs'

    checkpoint_file = checkpoint_dir / test_cfg.ckpt_file

    dm = NIHDataModule(
        dataset_path=data_cfg.dataset_path,
        phases=OmegaConf.create([dict(
            image_size=test_cfg.params.image_size,
            augment_rate=0,
            batch_size=test_cfg.params.batch_size,
            epoch_milestone=0
        )]),
        classes=test_cfg.classes
    )

    if hparams.architecture == 'eff_net_v2':
        model = EfficientNetV2Module.load_from_checkpoint(str(checkpoint_file))
    else:
        raise ValueError()

    trainer = Trainer(
        logger=ClearMLLogger(task, log_hyperparams=False),
        gpus=1,
        deterministic=True,
        default_root_dir=str(log_root_dir),
        weights_save_path=str(checkpoint_dir)
    )

    trainer.test(model, datamodule=dm)

    task.flush()


class NIHTestingLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument('config', type=str, help='Path to YAML config file.')

    def run(self, args) -> None:
        config = OmegaConf.load(args.config)

        test(config)


if __name__ == '__main__':
    NIHTestingLauncher(sys.argv[1:]).launch()
