"""Streamlit-based real-time monitoring dashboard.

Web-based UI with live camera feeds, detection stats, zone analytics,
alert history, and heatmap visualization.

Run with: streamlit run src/dashboard.py -- --config configs/default.yaml
"""

import time
from collections import deque

import cv2
import numpy as np
import streamlit as st

from src.analytics.heatmap import HeatmapGenerator
from src.models.registry import build_detector_from_config
from src.privacy import PrivacyFilter
from src.stream import VideoStream
from src.tracking.tracker import EnhancedTracker
from src.utils.drawing import draw_detections, draw_fps, draw_info, draw_tracks
from src.utils.fps import FPSCounter


def init_session_state():
    """Initialize session state defaults (runs once)."""
    defaults = {
        "running": False,
        "stream": None,
        "detector": None,
        "tracker": None,
        "heatmap_gen": None,
        "privacy_filter": None,
        "fps_counter": None,
        "frame_count": 0,
        "detection_log": deque(maxlen=100),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def stop_stream():
    """Release video stream and reset running state."""
    st.session_state.running = False
    if st.session_state.stream is not None:
        try:
            st.session_state.stream.release()
        except Exception:
            pass
        st.session_state.stream = None
    st.session_state.tracker = None
    st.session_state.heatmap_gen = None
    st.session_state.privacy_filter = None


def main():
    st.set_page_config(
        page_title="Object Detection Dashboard",
        page_icon="🎯",
        layout="wide",
    )

    init_session_state()

    st.title("Object Detection Dashboard")

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration")

    source_type = st.sidebar.selectbox("Source", ["Webcam", "IP Camera", "Video File"])

    if source_type == "Webcam":
        source_input = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0)
    elif source_type == "IP Camera":
        source_input = st.sidebar.text_input("Camera URL", "http://192.168.1.10:8080/video")
    else:
        source_input = st.sidebar.text_input("File Path", "")

    model_name = st.sidebar.selectbox(
        "Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]
    )
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

    st.sidebar.subheader("Features")
    enable_tracking = st.sidebar.checkbox("Object Tracking", value=True)
    enable_heatmap = st.sidebar.checkbox("Heatmap Overlay", value=False)
    enable_privacy = st.sidebar.checkbox("Privacy Mode", value=False)

    privacy_mode = "blur"
    if enable_privacy:
        privacy_mode = st.sidebar.selectbox("Privacy Filter", ["blur", "pixelate", "blackout"])

    # --- Start / Stop Controls ---
    col_start, col_stop = st.sidebar.columns(2)
    start_clicked = col_start.button("▶ Start", use_container_width=True)
    stop_clicked = col_stop.button("⏹ Stop", use_container_width=True)

    if stop_clicked:
        stop_stream()

    if start_clicked:
        # Stop any existing stream first
        stop_stream()

        # Resolve source
        source = int(source_input) if source_type == "Webcam" else source_input

        if source_type != "Webcam" and not source:
            st.sidebar.error("Please enter a valid source URL or file path.")
        else:
            try:
                with st.spinner("Connecting to video source..."):
                    st.session_state.stream = VideoStream(source=source)

                with st.spinner(f"Loading model {model_name}..."):
                    st.session_state.detector = build_detector_from_config(
                        {}, model_name=model_name, confidence=confidence
                    )

                st.session_state.fps_counter = FPSCounter()

                if enable_tracking:
                    st.session_state.tracker = EnhancedTracker()

                if enable_privacy:
                    st.session_state.privacy_filter = PrivacyFilter(mode=privacy_mode)

                st.session_state.frame_count = 0
                st.session_state.running = True

            except RuntimeError as e:
                st.sidebar.error(f"Failed to open source: {e}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")

    # --- Status ---
    st.sidebar.markdown("---")
    status = "🟢 Streaming" if st.session_state.running else "⚪ Stopped"
    st.sidebar.markdown(f"**Status:** {status}")
    st.sidebar.markdown(f"**Model:** {model_name}")
    st.sidebar.markdown(f"**Confidence:** {confidence:.0%}")

    # --- Main Layout ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Live Feed")
        video_placeholder = st.empty()

    with col2:
        st.subheader("Detection Stats")
        stats_placeholder = st.empty()

        st.subheader("Object Counts")
        counts_placeholder = st.empty()

    # Heatmap section
    heatmap_placeholder = None
    if enable_heatmap:
        st.subheader("Density Heatmap")
        heatmap_placeholder = st.empty()

    # --- Streaming Loop ---
    if not st.session_state.running:
        video_placeholder.info(
            "Click **Start** in the sidebar to begin streaming. "
            "Select your video source and model, then press Start."
        )
        return

    stream = st.session_state.stream
    detector = st.session_state.detector
    fps_counter = st.session_state.fps_counter
    tracker = st.session_state.tracker
    privacy_filter = st.session_state.privacy_filter

    if stream is None or detector is None:
        st.session_state.running = False
        return

    while st.session_state.running:
        if not stream.is_opened:
            st.session_state.running = False
            video_placeholder.warning("Video source ended or disconnected.")
            break

        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # Run detection (with or without tracking)
        if enable_tracking:
            detections = detector.detect_and_track(frame)
        else:
            detections = detector.detect(frame)

        # Update tracker
        if enable_tracking and tracker is not None:
            detections = tracker.update(detections, frame)

        # Apply privacy filter (before drawing boxes)
        if enable_privacy and privacy_filter is not None:
            frame = privacy_filter.apply(frame, detections)

        # Draw detection overlays
        draw_detections(
            frame, detections,
            show_confidence=True,
            show_track_id=enable_tracking,
        )

        # Draw tracking trails
        if enable_tracking and tracker is not None:
            trajectories = tracker.get_all_trajectories()
            draw_tracks(frame, trajectories)

        # Draw FPS and info
        fps_counter.tick()
        draw_fps(frame, fps_counter.fps)
        draw_info(frame, model_name, len(detections))

        # Update heatmap
        if enable_heatmap:
            if st.session_state.heatmap_gen is None:
                h, w = frame.shape[:2]
                st.session_state.heatmap_gen = HeatmapGenerator(frame_shape=(h, w))
            st.session_state.heatmap_gen.update(detections)

        # Convert BGR -> RGB and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update stats
        class_counts = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        with stats_placeholder.container():
            s1, s2 = st.columns(2)
            s1.metric("FPS", f"{fps_counter.fps:.1f}")
            s2.metric("Objects", len(detections))
            if detections:
                avg_conf = sum(d.confidence for d in detections) / len(detections)
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            if enable_tracking and tracker is not None:
                active = tracker._history.get_active_tracks()
                st.metric("Active Tracks", len(active))

        with counts_placeholder.container():
            if class_counts:
                for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                    st.text(f"{cls_name}: {count}")
            else:
                st.text("No objects detected")

        # Render heatmap
        if enable_heatmap and st.session_state.heatmap_gen is not None and heatmap_placeholder is not None:
            heatmap_frame = st.session_state.heatmap_gen.render(frame.copy())
            heatmap_rgb = cv2.cvtColor(heatmap_frame, cv2.COLOR_BGR2RGB)
            heatmap_placeholder.image(heatmap_rgb, channels="RGB", use_container_width=True)

        st.session_state.frame_count += 1

        # Small sleep to allow Streamlit to process stop button clicks
        time.sleep(0.03)


if __name__ == "__main__":
    main()
