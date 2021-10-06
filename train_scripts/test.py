import sys
import os
from argparse import ArgumentParser
from pathlib import Path

import clearml
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import Trainer

from data.nih_classification_data_module import NIHClassificationDataModule
from loggers.clearml_logger import ClearMLLogger
from modules.nih_efficient_net_v2_module import NIHEfficientNetV2Module
from modules.nih_resnet_module import NIHResNetModule
from utils.arg_launcher import ArgLauncher
from utils.misc import to_omega_conf


def test(cfg: DictConfig):
    pl.seed_everything(42)

    clearml.Task.force_requirements_env_freeze(requirements_file='requirements.txt')
    train_task = clearml.Task.get_task(task_id=cfg.task_id)

    task = clearml.Task.init(project_name=train_task.get_project_name(),
                             task_name=f'{train_task.name}_test',
                             auto_connect_frameworks=False,
                             output_uri=True)
    cluster_cfg = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.pop('cluster')), name='cluster_cfg'))
    cfg = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg), name='test_cfg'))

    task.execute_remotely(exit_process=True)

    hparams = to_omega_conf(train_task._get_configuration_dict('hparams'))
    data_cfg = to_omega_conf(train_task._get_configuration_dict('data_cfg'))

    log_root_dir = Path(os.path.expandvars(task.session.config.get('sdk.storage.log_dir'))).expanduser()
    in_log_root_dir = log_root_dir / train_task.task_id
    out_log_root_dir = log_root_dir / task.task_id

    checkpoint_dir = out_log_root_dir / 'checkpoints'
    checkpoint_file = in_log_root_dir / 'checkpoints' / cfg.ckpt_file

    dm = NIHClassificationDataModule(
        dataset_path=data_cfg.dataset_path,
        split_type=data_cfg.split_type,
        phases=OmegaConf.create([dict(
            image_size=cfg.params.image_size,
            augment_rate=0,
            batch_size=cfg.params.batch_size,
            epoch_milestone=0
        )]),
        classes=data_cfg.classes
    )

    if hparams.architecture == 'eff_net_v2':
        model_class = NIHEfficientNetV2Module
    elif hparams.architecture == 'resnet':
        model_class = NIHResNetModule
    else:
        raise ValueError()

    model = model_class.load_from_checkpoint(checkpoint_file.as_posix())

    # best_model_path = checkpoint_dir / (checkpoint_file.stem + '.pt')

    # model.save_to_file(best_model_path)
    # task.update_output_model(best_model_path.as_posix(), tags=['auroc_best'])

    trainer = Trainer(
        logger=ClearMLLogger(task, log_hyperparams=False),
        gpus=1,
        deterministic=True,
        default_root_dir=out_log_root_dir.as_posix(),
        weights_save_path=checkpoint_dir.as_posix()
    )

    trainer.test(model, datamodule=dm)

    task.flush()


class NIHTestingLauncher(ArgLauncher):

    def setup_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument('config_dir', type=str, help='Path to directory with YAML config files.')

    def run(self, args) -> None:
        cfg_root = Path(args.config_dir)

        config = OmegaConf.load(cfg_root / 'test_config.yaml')
        config.cluster = OmegaConf.load(cfg_root / 'test_cluster.yaml')

        test(config)


if __name__ == '__main__':
    NIHTestingLauncher(sys.argv[1:]).launch()
