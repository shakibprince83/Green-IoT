# Supervisor demo

From the repository root:

```bash
python -m pip install -e '.[dev]'
greenbench doctor
greenbench protocol validate
greenbench model audit --framework all
greenbench study smoke
greenbench schedule create --profile pilot
pytest
```

Explain that unavailable frameworks/hardware are reported, never faked. Show `protocols/core_v1.yaml`, the persisted schedule, canonical initialization hash, schema guard against Tier-3 energy, and the dashboard empty state. On the experiment machine, prepare data and install both framework environments before `greenbench study pilot`.
