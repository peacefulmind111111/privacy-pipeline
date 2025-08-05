"""Streamlit dashboard for DP experiment analytics.

This prototype app visualizes JSON reports produced by the unified
``privacy_pipeline`` experiments.  The interface supports filtering by
experiment name and plotting key metrics.  Per‑iteration diagnostics such as
loss, training accuracy, and privacy budget consumption are displayed as
interactive charts.

Usage
-----
To run the app locally, first ensure that Streamlit is installed::

    pip install streamlit pandas

Then launch the app from the command line::

    streamlit run streamlit_app.py

By default the app looks for JSON files in the ``outputs`` directory relative
to this file.  Adjust :data:`RESULTS_DIR` if your results live elsewhere.
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any

import pandas as pd  # type: ignore
import streamlit as st  # type: ignore

# Directory containing JSON result files.  Modify this path as needed.
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def load_results(directory: str) -> List[Dict[str, Any]]:
    """Load all JSON result files from a directory.

    Each JSON file is expected to follow the schema produced by
    :class:`privacy_pipeline.logger.ExperimentLogger`.
    """
    results: List[Dict[str, Any]] = []
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue
        data["_filename"] = filename
        results.append(data)
    return results


def main() -> None:
    st.title("Differentially Private ML Experiment Dashboard")
    st.markdown(
        "This dashboard visualizes results from various DP experiments. "
        "Use the sidebar to filter experiments and choose which metrics to display."
    )

    # Load results
    if not os.path.exists(RESULTS_DIR):
        st.error(f"Results directory not found: {RESULTS_DIR}")
        return
    raw_results = load_results(RESULTS_DIR)
    if not raw_results:
        st.warning(
            "No result files were found. Please generate experiments and place "
            "their JSON outputs in the outputs directory."
        )
        return

    # Build summary DataFrame
    summary_records = []
    for data in raw_results:
        record: Dict[str, Any] = {
            "experiment_name": data.get("experiment_name", data.get("_filename")),
            "timestamp": data.get("timestamp"),
        }
        hyperparams = data.get("hyperparameters", {})
        final_metrics = data.get("final_metrics", {})
        for k, v in hyperparams.items():
            record[f"hp_{k}"] = v
        for k, v in final_metrics.items():
            record[f"metric_{k}"] = v
        summary_records.append(record)

    df = pd.DataFrame(summary_records)

    # Sidebar filters
    st.sidebar.header("Filters")

    exp_substring = st.sidebar.text_input(
        "Experiment name contains", value="", help="Leave blank to include all experiments."
    )
    if exp_substring:
        df = df[df["experiment_name"].str.contains(exp_substring, case=False)]

    metric_columns = [c for c in df.columns if c.startswith("metric_")]
    if not metric_columns:
        st.error("No metric columns found in results.")
        return
    selected_metric = st.sidebar.selectbox("Metric to plot", metric_columns, index=0)

    st.subheader("Filtered Results")
    st.dataframe(df)

    st.subheader(f"{selected_metric} across experiments")
    if not df.empty:
        chart_data = df[["experiment_name", selected_metric]].set_index("experiment_name")
        st.bar_chart(chart_data)
    else:
        st.info("No experiments match the current filters.")

    st.subheader("Per-iteration metrics")
    exp_options = [d.get("experiment_name", d.get("_filename")) for d in raw_results]
    selected_exp = st.selectbox("Experiment", exp_options)
    selected_data = next(d for d in raw_results if d.get("experiment_name", d.get("_filename")) == selected_exp)
    iter_df = pd.DataFrame(selected_data.get("iterations", []))
    if not iter_df.empty:
        st.line_chart(iter_df.set_index("iteration")[["loss", "train_accuracy", "epsilon"]])
    else:
        st.info("Selected experiment has no per-iteration data.")


if __name__ == "__main__":
    main()
