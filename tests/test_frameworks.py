import importlib.util

import pytest

from greenbench.models import ADAM, build_pytorch, build_tensorflow


def test_optimizer_explicit():
    assert ADAM == {
        "learning_rate": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-7,
        "amsgrad": False,
        "weight_decay": 0.0,
    }


@pytest.mark.framework
@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_pytorch_model():
    import torch

    model = build_pytorch()
    assert sum(p.numel() for p in model.parameters()) == 1_199_882
    assert tuple(model(torch.zeros(2, 1, 28, 28)).shape) == (2, 10)


@pytest.mark.framework
@pytest.mark.skipif(
    importlib.util.find_spec("tensorflow") is None, reason="TensorFlow not installed"
)
def test_tensorflow_model():
    import tensorflow as tf

    model = build_tensorflow()
    assert model.count_params() == 1_199_882
    assert tuple(model(tf.zeros((2, 28, 28, 1))).shape) == (2, 10)
