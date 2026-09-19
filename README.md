# GREEN AI BENCHMARK

## A Controlled Comparison of Equivalent Workloads across PyTorch & TensorFlow

GREEN AI BENCHMARK is a reproducible research system for comparing predictive quality, runtime, energy, carbon estimates, and resource use under controlled machine-learning workloads. Its primary framework experiment uses numerically identical initial weights, deterministic sample orders, explicit optimizer semantics, and paired runs of a matched CNN in PyTorch and TensorFlow.

No framework is assumed to be greener. The repository produces evidence; it does not contain invented results.

### Study design

- `core_algorithm`: classical and CNN-family comparisons; answers a model-family question.
- `core_framework`: matched PyTorch/TensorFlow CNN on MNIST and Fashion-MNIST; the primary framework-effect study.
- `extension_*`: architecture, operator, device, batch, precision and optimized studies; never pooled into primary analysis.

The locked CNN is `Conv(1,32,3)-ReLU-Conv(32,64,3)-ReLU-MaxPool(2)-Flatten-Dense(128)-ReLU-Dense(10)` with exactly **1,199,882** trainable parameters. Primary settings are Adam (`lr=0.001`, `beta1=0.9`, `beta2=0.999`, `epsilon=1e-7`, no AMSGrad or weight decay), batch size 64, 10 epochs, FP32, logits cross-entropy, and no augmentation or early stopping.

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
greenbench doctor
greenbench protocol validate
greenbench data prepare --dataset mnist
greenbench study smoke
```

Use separate framework environments if their GPU dependencies conflict:

```bash
python -m pip install -e '.[pytorch]'
# or in another environment
python -m pip install -e '.[tensorflow]'
```

### Research commands

```bash
greenbench doctor
greenbench protocol validate
greenbench schedule create --profile pilot
greenbench model audit --framework all
greenbench study pilot
greenbench study run --profile official --confirm-official
greenbench study resume --profile official --confirm-official
greenbench results validate
greenbench results aggregate
greenbench analyze stats
greenbench analyze green
greenbench report build
greenbench dashboard
greenbench reproduce
```

`SMOKE` is only a correctness check. `PILOT` uses real but limited measurements. `OFFICIAL` is protocol-locked and immutable. Framework runs fail preflight if equivalence cannot be proved.

### Measurement validity

| Tier | Meaning | Examples |
|---|---|---|
| 1 | Direct hardware measurement | RAPL, NVML integration, external meter |
| 2 | Software estimate | CodeCarbon, CarbonTracker |
| 3 | Resource proxy only | runtime, RAM, utilization, temperature |

Tier 3 is never labelled energy. Carbon values are explicitly estimates. Training/inference, warm-up/measured work, and compute-only/end-to-end modes remain separate.

### Outputs

- `results/raw/`: immutable run JSON
- `results/telemetry/`: timestamped raw telemetry
- `results/processed/`: regenerable analysis
- `results/tables/`, `results/figures/`: generated publication material
- `artifacts/`: hashed splits, schedules, initializations and environments

See [research protocol](docs/RESEARCH_PROTOCOL.md), [fairness controls](docs/FAIRNESS_CONTROLS.md), [measurement validity](docs/MEASUREMENT_VALIDITY.md), [reproducibility](docs/REPRODUCIBILITY.md), and [supervisor demo](docs/SUPERVISOR_DEMO.md).

### Responsible interpretation

Conclusions apply only to tested workloads, versions, devices, and measurement tiers. Raw metrics, paired effects, confidence intervals and Pareto trade-offs are primary. GES is secondary decision support. Classical CPU versus GPU CNN results are system-level comparisons.

MIT licensed. Cite with `CITATION.cff`.
