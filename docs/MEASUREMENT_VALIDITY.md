# Measurement validity

Tier 1 is hardware measurement (RAPL, NVML integration or external meter). Tier 2 is software estimation (CodeCarbon/CarbonTracker). Tier 3 contains resource proxies only. The run schema rejects joules or watts attached to Tier 3 records.

NVML power samples, when supported, are timestamped and integrated by trapezoidal integration. Gross energy is primary; an idle-adjusted result is sensitivity-only. Carbon is always an estimate with intensity and source. Inference excludes warm-up and distinguishes compute-only from end-to-end scope.

Run order is paired/counterbalanced. Cooling, idle sampling and temperature thresholds are machine-specific configuration. The scheduler is persisted before execution, guarded by a lock, and resumable.
