# Fairness controls

The canonical parameter representation uses PyTorch-style OIHW convolution and output-by-input dense layouts. TensorFlow receives transposed HWIO and input-by-output tensors; round-trip equality is tested. A stored SHA-256 hash identifies each initialization.

Core preflight checks parameter count, output shape, initialization hash, split hash, batch-order hash, optimizer fields, precision, logits loss and execution modes. PyTorch and TensorFlow use eager execution in the primary study; compilation, XLA, AMP and mixed precision are disabled. Accelerated/native modes are extensions.

The test set is untouched until the selected configuration is final. Test indices cannot be passed to selection APIs. Failed or excluded runs remain recorded with reasons.
