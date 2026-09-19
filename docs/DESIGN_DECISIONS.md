# Design decisions

1. Framework-neutral NumPy tensors, not same-seed assumptions, define initialization parity.
2. A minimal dependency core keeps doctor, protocol, statistics and integrity checks usable without GPU frameworks.
3. Framework adapters are optional because TensorFlow/PyTorch GPU dependency sets may conflict.
4. GES is secondary; raw measurements, paired effects, confidence intervals and Pareto fronts remain primary.
5. No demo measurement file is placed in official-result paths.
