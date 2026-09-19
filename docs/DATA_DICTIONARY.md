# Data dictionary

`RunRecord` in `src/greenbench/provenance.py` is the authoritative typed schema. `classification` separates SMOKE, PILOT, OFFICIAL and EXTENSION. `measurement_tier`, `measurement_backend`, `measurement_method` and `measurement_scope` prevent incompatible measurements from being pooled. Hash fields identify protocol, config, hardware, architecture, initialization, dataset and split. `run_status`, `excluded` and `exclusion_reason` preserve failures and exclusions.
