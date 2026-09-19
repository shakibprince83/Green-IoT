from __future__ import annotations

from typing import Any

from .initialization import canonical_weights, parameter_count, to_tensorflow

ADAM = {
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1e-7,
    "amsgrad": False,
    "weight_decay": 0.0,
}


def build_pytorch(seed: int = 11) -> Any:
    import torch

    class MatchedCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, 3, bias=True)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, bias=True)
            self.pool = torch.nn.MaxPool2d(2)
            self.dense1 = torch.nn.Linear(9216, 128, bias=True)
            self.dense2 = torch.nn.Linear(128, 10, bias=True)

        def forward(self, x: Any) -> Any:
            x = torch.relu(self.conv1(x))
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            return self.dense2(torch.relu(self.dense1(x)))

    model = MatchedCNN()
    weights = canonical_weights(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.from_numpy(weights[name]))
    assert sum(x.numel() for x in model.parameters()) == parameter_count()
    return model


def build_tensorflow(seed: int = 11) -> Any:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding="valid", use_bias=True, name="conv1")(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="valid", use_bias=True, name="conv2")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, use_bias=True, name="dense1")(x)
    x = tf.keras.layers.ReLU()(x)
    outputs = tf.keras.layers.Dense(10, use_bias=True, name="dense2")(x)
    model = tf.keras.Model(inputs, outputs, name="matched_cnn")
    weights = to_tensorflow(canonical_weights(seed))
    for layer in (
        model.get_layer("conv1"),
        model.get_layer("conv2"),
        model.get_layer("dense1"),
        model.get_layer("dense2"),
    ):
        layer.set_weights([weights[f"{layer.name}.kernel"], weights[f"{layer.name}.bias"]])
    assert model.count_params() == parameter_count()
    return model


def audit(framework: str) -> dict[str, Any]:
    result: dict[str, Any] = {"expected_parameters": parameter_count(), "optimizer": ADAM}
    if framework in {"pytorch", "all"}:
        try:
            model = build_pytorch()
            result["pytorch"] = {
                "available": True,
                "parameters": sum(p.numel() for p in model.parameters()),
                "output": [1, 10],
            }
        except ImportError:
            result["pytorch"] = {"available": False, "reason": "PyTorch not installed"}
    if framework in {"tensorflow", "all"}:
        try:
            model = build_tensorflow()
            result["tensorflow"] = {
                "available": True,
                "parameters": model.count_params(),
                "output": [1, 10],
            }
        except ImportError:
            result["tensorflow"] = {"available": False, "reason": "TensorFlow not installed"}
    return result
