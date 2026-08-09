import copy

from .nanodet_plus_head import NanoDetPlusHead
from .nanodet_plus_softmax_head import NanoDetPlusSoftmaxHead
from .simple_conv_head import SimpleConvHead


def build_head(cfg):
    head_cfg = copy.deepcopy(cfg)
    name = head_cfg.pop("name")
    if name == "NanoDetPlusHead":
        return NanoDetPlusHead(**head_cfg)
    elif name == "NanoDetPlusSoftmaxHead":
        return NanoDetPlusSoftmaxHead(**head_cfg)
    elif name == "SimpleConvHead":
        return SimpleConvHead(**head_cfg)
    else:
        raise NotImplementedError(f"Head {name} not supported. Use NanoDetPlusHead, NanoDetPlusSoftmaxHead, or SimpleConvHead.")
