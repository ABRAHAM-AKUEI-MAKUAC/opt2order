from .mlp import build_mnist_mlp
from .cnn import build_cifar_vgg
from .rnn import build_addition_lstm

MODEL_BUILDERS = {
    "mnist_mlp": build_mnist_mlp,
    "cifar_vgg": build_cifar_vgg,
    "addition_lstm": build_addition_lstm,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Known: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](**kwargs)


__all__ = ["build_model", "MODEL_BUILDERS"]
