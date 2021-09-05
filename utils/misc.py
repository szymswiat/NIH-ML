from typing import Union, Any

from omegaconf import OmegaConf, DictConfig, ListConfig


def to_omega_conf(
        cfg: Union[dict, list]
) -> Union[DictConfig, ListConfig]:
    def convert(c: Union[dict, list, Any]) -> Union[dict, list, Any]:

        if isinstance(c, dict):
            return {k: convert(v) for k, v in c.items()}
        elif isinstance(c, list):
            return [convert(v) for v in c]
        else:
            return c

    return OmegaConf.create(convert(cfg))
