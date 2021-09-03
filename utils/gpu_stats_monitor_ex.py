import time
from typing import Any, List

from pytorch_lightning.callbacks import GPUStatsMonitor
from pytorch_lightning.utilities import rank_zero_only


class GPUStatsMonitorEx(GPUStatsMonitor):

    def __init__(
            self,
            memory_utilization: bool = True,
            gpu_utilization: bool = True,
            intra_step_time: bool = False,
            inter_step_time: bool = False,
            fan_speed: bool = False,
            temperature: bool = False,
            prog_bar_filters: List[str] = []
    ):
        super().__init__(memory_utilization, gpu_utilization, intra_step_time, inter_step_time, fan_speed, temperature)

        self._prog_bar_filters = prog_bar_filters

    @rank_zero_only
    def on_train_batch_end(
            self, trainer, pl_module, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if not self._should_log(trainer):
            return

        gpu_stat_keys = self._get_gpu_stat_keys() + self._get_gpu_device_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(self._gpu_ids, gpu_stats, gpu_stat_keys)

        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs['batch_time/intra_step (ms)'] = (time.time() - self._snap_intra_step_time) * 1000

        for k, v in logs.items():
            for filter_key in self._prog_bar_filters:
                if filter_key in k:
                    pl_module.log(k, v, prog_bar=True, logger=False, on_step=True, on_epoch=False)

        trainer.logger.log_metrics(logs, step=trainer.global_step)
