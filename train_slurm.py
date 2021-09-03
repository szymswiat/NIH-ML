import sys

from omegaconf import DictConfig
from test_tube import SlurmCluster

from train import train
from utils.slurm.slurm_launcher import SLURMLauncher


class NIHTrainingLauncher(SLURMLauncher):

    def run(self, cfg: DictConfig, cluster: SlurmCluster) -> None:
        train(cfg, is_hpc_exp=True)


if __name__ == '__main__':
    NIHTrainingLauncher(sys.argv[1:]).launch()
