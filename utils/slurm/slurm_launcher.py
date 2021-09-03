from abc import abstractmethod
from argparse import ArgumentParser

from omegaconf import OmegaConf, ListConfig, DictConfig
from test_tube import HyperOptArgumentParser

from utils.slurm.slurm_cluster_ext import SlurmClusterExt


class SLURMLauncher:

    def __init__(self, args):
        self._args = args
        self._parser = HyperOptArgumentParser()

    def setup_params(self, parser: ArgumentParser) -> None:
        """
        Override to setup arbitrary CLI args.
        :param parser: Argument parser.
        :return: None
        """
        parser.add_argument('-c', '--config', type=str, required=True,
                            help='Path to YAML config file.')

    def setup_hparams(self, parser: HyperOptArgumentParser, hyperparam_opts: ListConfig, strategy: str) -> int:
        """
        Override this method to add hyperparams for grid search.
        See example below.
        @param parser: Argument parser.
        @param hyperparam_opts: Hyperparam options dict from yaml config file.
        @param strategy: Search strategy
        @return: Returns amount of generated slurm tasks, based on count of hyperparams combinations.
        """
        all_opt_count = 1

        for hp_data in hyperparam_opts:
            name = f"--hyp_opt__{hp_data.name}"

            if 'value_range' in hp_data:
                low, high = hp_data.value_range
                parser.opt_range(name, tunable=True, low=low, high=high, type=type(low))
                assert strategy != 'grid_search'
            elif 'value_list' in hp_data:
                v = hp_data.value_list
                parser.opt_list(name, tunable=True, options=v, type=type(v[0]))
                all_opt_count *= len(v)
            else:
                raise ValueError('Unsupported search strategy.')

        return all_opt_count

    def configure_cluster(self, cluster_cfg: DictConfig, cluster: SlurmClusterExt) -> None:
        """
        Configures SLURM cluster resources allocation.
        :param cluster_cfg: Cluster params.
        :param cluster: Cluster object.
        :return: None
        """
        # configure cluster
        cluster.per_experiment_nb_nodes = cluster_cfg.nodes_per_exp
        cluster.per_experiment_nb_gpus = cluster_cfg.gpus_per_node
        cluster.memory_mb_per_node = cluster_cfg.mem_per_node * 1024
        cluster.per_experiment_nb_cpus = cluster_cfg.cpus_per_node
        cluster.minutes_to_checkpoint_before_walltime = 1
        cluster.job_time = cluster_cfg.time

        # cluster.add_slurm_cmd(cmd='ntasks-per-node', value=1, comment='1 task per GPU')
        cluster.add_slurm_cmd(cmd='constraint', value='localfs', comment='Enable local filesystem.')
        cluster.add_slurm_cmd(cmd='partition', value=cluster_cfg.partition,
                              comment='Use partition dedicated for GPUs.')

    @abstractmethod
    def run(self, cfg: DictConfig, cluster: SlurmClusterExt) -> None:
        """
        Override to execute arbitrary code with given hparams in SLURM environment.
        :param cfg: Dictionary with configuration.
        :param cluster: SlurmClusterExt object.
        :return: None
        """
        raise NotImplementedError()

    def launch(self) -> None:
        """
        Runs launcher.
        @return: None
        """
        self.setup_params(self._parser)
        args, _ = self._parser.parse_known_args(self._args)

        cfg = OmegaConf.load(args.config)

        hparam_config = cfg.hparams_opt
        if hparam_config.enabled:
            strategy = hparam_config.strategy
            if strategy:
                self._parser.strategy = strategy
                grid_trials = self.setup_hparams(self._parser, hparam_config.opts, strategy)
                if 'grid' in strategy:
                    nb_trials = grid_trials
                elif 'random' in strategy:
                    nb_trials = hparam_config.trials
                else:
                    raise ValueError()
            else:
                nb_trials = 1
        else:
            nb_trials = 1

        args = self._parser.parse_args(self._args)

        cluster = SlurmClusterExt(
            hyperparam_optimizer=args,
            log_path='./slurm_logs'
        )
        self.configure_cluster(cfg.cluster, cluster)

        # submit a script
        cluster.optimize_parallel_cluster_ext(
            self.run,
            config=cfg,
            nb_trials=nb_trials,
            job_name=cfg.cluster.job_name
        )
