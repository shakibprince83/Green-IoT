# Research protocol

The primary claim is limited to the effect of framework implementation for a locked, matched CNN workload on MNIST and Fashion-MNIST. Each dataset/seed is a pair. Both members use the same official source data, saved split indices, serialized epoch order, canonical NumPy initialization, explicit Adam parameters, FP32 logits loss, batch size 64 and 10 epochs.

Five paired seeds are the minimum final sample. Framework order is randomized within every pair and persisted before execution. Training and inference are separate phases. The primary predictive metric is Macro-F1; secondary metrics are accuracy, macro precision/recall, OvR macro ROC-AUC, MCC and log loss. Analysis reports original-unit paired differences, bootstrap CIs, corrected p-values, effect sizes and practical thresholds. Optional TOST uses a preregistered Macro-F1 margin and is not primary.

Classical-model, architecture, operator, device, batch, precision and native-optimization work is stored under separate experiment-family labels and cannot enter the primary paired analysis.
