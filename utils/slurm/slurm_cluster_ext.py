from __future__ import annotations
from typing import Callable, Dict

import os
import signal
import traceback

from omegaconf import DictConfig
from test_tube import SlurmCluster


class SlurmClusterExt(SlurmCluster):

    def optimize_parallel_cluster_ext(
            self,
            train_function: Callable[[DictConfig, SlurmClusterExt], None],
            config: DictConfig,
            nb_trials: int,
            job_name: str,
            enable_auto_resubmit=False,
            job_display_name=None,
            on_gpu=True,
    ):

        self.job_name = job_name
        self.job_display_name = job_display_name if job_display_name else job_name
        self.on_gpu = on_gpu
        self.enable_auto_resubmit = enable_auto_resubmit

        # layout logging structure
        self._SlurmCluster__layout_logging_dir()

        if self.is_from_slurm_object:
            # Script is called by slurm: it's an actual experiment.
            self._run_experiment_ext(train_function, config)
        else:
            # Launcher script. Generate trials and launch jobs.

            # generate hopt trials
            trials = self.hyperparam_optimizer.generate_trials(nb_trials)

            # get the max test tube exp version so far if it's there
            scripts_path = os.path.join(self.log_path, 'slurm_out_logs')
            next_trial_version = self._SlurmCluster__get_max_trial_version(scripts_path)

            # for each trial, generate a slurm command
            for i, trial_params in enumerate(trials):
                exp_i = i + next_trial_version
                self.schedule_experiment(trial_params, exp_i)

    def _run_experiment_ext(
            self,
            train_function: Callable[[DictConfig, SlurmClusterExt], None],
            config: DictConfig
    ):
        if self.enable_auto_resubmit:
            print('setting signal')
            signal.signal(signal.SIGUSR1, self.sig_handler)
            signal.signal(signal.SIGTERM, self.term_handler)

        try:
            for opt_name, opt_value in self.hyperparam_optimizer.__dict__.items():
                hparams = config.hparams
                if opt_name.startswith('hyp_opt__'):
                    opt_name_parts = opt_name.replace('hyp_opt__', '').split('.')
                    if len(opt_name_parts) > 1:
                        for opt_name_part in opt_name_parts[:-1]:
                            hparams = hparams[opt_name_part]
                    hparams[opt_name_parts[-1]] = opt_value

            config.cluster.hpc_exp_number = self.hyperparam_optimizer.hpc_exp_number

            # run training
            train_function(config, self)

        except Exception as e:
            print('Caught exception in worker thread', e)

            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            raise SystemExit
