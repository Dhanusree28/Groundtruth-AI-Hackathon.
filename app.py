"""
Streamlit UI for the Automated Insight Engine.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.pipeline.engine import InsightEngine

st.set_page_config(
    page_title="Automated Insight Engine",
    page_icon="📊",
    layout="wide",
)

ENGINE = InsightEngine()

st.title("📊 Automated Insight Engine")
st.caption(
    "Upload blended marketing funnels, train an ML model, and export ready-to-share reports."
)

with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    st.info(
        "No file? We'll fall back to the bundled `sample_kag_conversion.csv` dataset.",
        icon="ℹ️",
    )

    st.header("2. Report Format")
    report_type = st.radio("Choose output", options=["PDF", "PPTX"], horizontal=False)

    st.header("3. Optional Settings")
    st.checkbox("Use uploaded file only", key="use_uploaded_only", value=True, disabled=True)

placeholder = st.empty()

def persist_upload(upload) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    saved_path = ENGINE.paths.uploads_dir / f"uploaded_{timestamp}.csv"
    saved_path.write_bytes(upload.getbuffer())
    return saved_path


def render_metrics(metrics: dict) -> None:
    cols = st.columns(len(metrics))
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(label=name.title(), value=value)


if st.button("Run Insight Engine", type="primary"):
    data_path = None
    if uploaded_file is not None:
        data_path = persist_upload(uploaded_file)

    with st.spinner("Training model, crunching insights, and building the report..."):
        outputs = ENGINE.run(file_path=data_path, report_type=report_type)

    st.success("Done! Your assets are ready.")
    render_metrics(outputs["metrics"])

    st.subheader("Narrative Summary")
    st.write(outputs["narrative"])

    report_path = outputs["report"]
    st.download_button(
        label=f"Download {report_type}",
        data=report_path.read_bytes(),
        file_name=report_path.name,
        mime="application/pdf" if report_type == "PDF" else "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )

    with st.expander("Model artifacts"):
        st.download_button(
            label="Download trained model (.joblib)",
            data=Path(outputs["model"]).read_bytes(),
            file_name="conversion_model.joblib",
            mime="application/octet-stream",
        )
        st.download_button(
            label="Download evaluation sample (.csv)",
            data=Path(outputs["evaluation_sample"]).read_bytes(),
            file_name="evaluation_sample.csv",
            mime="text/csv",
        )
else:
    placeholder.info(
        "Configure the report in the sidebar and click **Run Insight Engine** to generate PDF/PPTX output."
    )

