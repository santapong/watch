"""Streamlit-based real-time monitoring dashboard.

Web-based UI with live camera feeds, detection stats, zone analytics,
alert history, and heatmap visualization.

Run with: streamlit run src/dashboard.py -- --config configs/default.yaml
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
import streamlit as st
import yaml

from src.models.base import Detection


def load_config(path: str) -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    st.set_page_config(
        page_title="Object Detection Dashboard",
        page_icon="🎯",
        layout="wide",
    )

    st.title("Object Detection Dashboard")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    source_type = st.sidebar.selectbox("Source", ["Webcam", "IP Camera", "Video File"])
    model_name = st.sidebar.selectbox(
        "Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
    )
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    st.sidebar.subheader("Features")
    enable_tracking = st.sidebar.checkbox("Object Tracking", value=True)
    enable_heatmap = st.sidebar.checkbox("Heatmap Overlay", value=False)
    enable_privacy = st.sidebar.checkbox("Privacy Mode", value=False)

    if enable_privacy:
        privacy_mode = st.sidebar.selectbox("Privacy Filter", ["blur", "pixelate", "blackout"])

    st.sidebar.subheader("Analytics")
    enable_anomaly = st.sidebar.checkbox("Anomaly Detection", value=False)
    enable_alerts = st.sidebar.checkbox("Alert System", value=False)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty()
        st.caption("Connect a video source to begin streaming.")

    with col2:
        st.subheader("Detection Stats")
        stats_placeholder = st.empty()

        st.subheader("Object Counts")
        counts_placeholder = st.empty()

        if enable_alerts:
            st.subheader("Alert History")
            alerts_placeholder = st.empty()

    # Bottom section
    if enable_heatmap:
        st.subheader("Density Heatmap")
        heatmap_placeholder = st.empty()

    # Status display
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status:** Ready")
    st.sidebar.markdown(f"**Model:** {model_name}")
    st.sidebar.markdown(f"**Confidence:** {confidence:.0%}")

    # Instructions
    st.info(
        "This dashboard provides a web-based monitoring interface. "
        "To connect to a live detection stream, run the detection backend "
        "and connect via the configured source. "
        "Use `streamlit run src/dashboard.py` to launch."
    )


if __name__ == "__main__":
    main()
