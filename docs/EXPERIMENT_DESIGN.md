# Experiment design

The unit of analysis is the within-seed framework pair. A schedule randomizes which framework runs first for every seed and dataset. Official analysis requires all partners; incomplete pairs are rejected. Runs use fixed epochs rather than early stopping. Validation selects configurations; official test data are evaluated only after selection.

Primary outcomes are training energy (when Tier 1 or 2 supports it), training time, Macro-F1 and inference energy/latency. Secondary outcomes include other predictive metrics, memory, utilization and carbon estimates. GES and TOST are secondary/sensitivity analyses. Architecture, operator, device, batch-size, precision and optimized modes are exploratory extensions.
