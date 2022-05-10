from .model_interface import ModelInterface
import yaml
import copy


# Helper function for (deep) copy-and-update of a dict
def _deepcopy_dict(d: dict, update: dict = {}) -> dict:
    out = {}
    # Loop over d
    for k, v in d.items():
        if isinstance(v, dict):
            # If k maps to a dict, recurse
            out[k] = _deepcopy_dict(v, update.get(k, {}))
        else:
            # Otherwise, out[k] = update[k] (if it exists) or a (shallow) copy of v otherwise
            out[k] = update.get(k, copy.copy(v))
    # Loop over udpate to get any missing keys
    for k, v in update.items():
        if k not in out:
            out[k] = v
    return out


def load_from_config(base_config, config: dict) -> ModelInterface:
    with open(base_config, 'r') as f:
        base_config = yaml.safe_load(f)

    ckpt = config.pop('model_path', None)
    kwarg_hparams = _deepcopy_dict(base_config, update=config)
    if ckpt is not None:
        return ModelInterface.load_from_checkpoint(ckpt, map_location='cpu', **kwarg_hparams)
    else:
        return ModelInterface(**kwarg_hparams)


__all__ = ['ModelInterface', 'load_from_config']
