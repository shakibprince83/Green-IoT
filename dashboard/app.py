import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="GREEN AI BENCHMARK", layout="wide")
st.title("GREEN AI BENCHMARK")
st.caption("A Controlled Comparison of Equivalent Workloads across PyTorch & TensorFlow")
root = Path(__file__).resolve().parents[1]
records = []
for path in (root / "results" / "raw").rglob("*.json"):
    records.append(json.loads(path.read_text(encoding="utf-8")))
if not records:
    st.info(
        "No measured runs exist yet. Run a pilot after doctor, protocol and model "
        "preflight checks pass. No synthetic values are displayed."
    )
else:
    official = [r for r in records if r.get("classification") == "OFFICIAL"]
    st.metric("Validated run files", len(records))
    st.metric("Official runs", len(official))
    st.dataframe(records, use_container_width=True)
