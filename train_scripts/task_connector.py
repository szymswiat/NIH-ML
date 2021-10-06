import clearml
from omegaconf import DictConfig
from omegaconf import OmegaConf

from utils.misc import to_omega_conf


def connect_with_task(
        cfg: DictConfig,
        project_name: str,
        run_offline=False
) -> clearml.Task:
    task_args = dict(auto_connect_frameworks=False,
                     output_uri=True)

    if cfg.training.restore_training.enabled:
        task_args.update(dict(continue_last_task=cfg.training.restore_training.task_id))
    else:
        task_args.update(dict(project_name=project_name,
                              task_name=cfg.task_name))

    if run_offline:
        clearml.Task.set_offline(True)

    clearml.Task.force_requirements_env_freeze(requirements_file='requirements.txt')
    task = clearml.Task.init(**task_args)

    cfg.cluster = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.cluster), name='cluster_cfg'))
    cfg.hparams = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.hparams), name='hparams'))
    cfg.data = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.data), name='data_cfg'))
    cfg.training = to_omega_conf(task.connect_configuration(OmegaConf.to_object(cfg.training), name='training_cfg'))

    if not run_offline:
        task.execute_remotely(exit_process=True)

    return task
