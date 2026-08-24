from .sgdm import SGDMomentum
from .adam import Adam
from .lbfgs import LBFGS
from .hessian_free import HessianFree
from .kfac import KFAC
from .newton import DampedNewton

OPTIMIZER_REGISTRY = {
    "sgdm": SGDMomentum,
    "adam": Adam,
    "lbfgs": LBFGS,
    "hf": HessianFree,
    "hessian_free": HessianFree,
    "kfac": KFAC,
    "newton": DampedNewton,
}


def build_optimizer(name: str, params, **kwargs):
    """Instantiate an optimizer by name. Used by the training harness."""
    key = name.lower()
    if key not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. "
                         f"Known: {sorted(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[key](params, **kwargs)


__all__ = ["SGDMomentum", "Adam", "LBFGS", "HessianFree", "KFAC",
           "DampedNewton", "build_optimizer", "OPTIMIZER_REGISTRY"]
