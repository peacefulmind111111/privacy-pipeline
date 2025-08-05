import glob
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from experiment_utils import load_results

st.title("DP Experiment Dashboard")

# Load experiment results
results = load_results("outputs/*.json")
if not results:
    st.info("No experiment results found. Run experiments to generate JSON outputs.")
else:
    records = [r.to_json() for r in results.values()]
    df = pd.DataFrame(records)
    st.dataframe(df)
    metric_cols = [col for col in df.columns if col not in {"params", "metrics"}]
    st.write("Summary Metrics")
    metrics_df = pd.json_normalize(df["metrics"])
    st.dataframe(pd.concat([df[metric_cols], metrics_df], axis=1))
