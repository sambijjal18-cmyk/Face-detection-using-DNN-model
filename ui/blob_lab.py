"""Tab 3 — BLOB Lab: how the SSD input blob varies face-to-face (DNN path)."""

from __future__ import annotations

import numpy as np
import streamlit as st

from face import config, signals

from .common import Camera, bgr_to_rgb, missing_models_warning


def _stats_table(stats):
    return [
        {"channel": s.name, "min": round(s.min, 1), "max": round(s.max, 1),
         "mean": round(s.mean, 2), "std": round(s.std, 2)}
        for s in stats
    ]


def render() -> None:
    st.header("BLOB Lab — the DNN's actual input signal")
    st.caption("The SSD never sees your raw frame. It sees blobFromImage(): resized to "
               "300×300 with the BGR mean (104, 177, 123) subtracted. Those mean-subtracted "
               "channels shift with every different face and lighting condition.")
    st.code(
        "blob = cv2.dnn.blobFromImage(resize(frame, (300,300)), 1.0, (300,300), (104,177,123))",
        language="python",
    )

    src = st.radio("Source", ["Snapshot from webcam", "Upload image"], horizontal=True, key="blob_src")
    frame = None
    if src == "Snapshot from webcam":
        cam_index = st.number_input("Camera index", 0, 8, 0, key="blob_cam")
        if st.button("Capture frame", key="blob_capture"):
            with Camera(int(cam_index)) as cam:
                frame = cam.read()
            if frame is not None:
                st.session_state["blob_frame"] = frame
        frame = st.session_state.get("blob_frame")
    else:
        up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp"], key="blob_upload")
        if up is not None:
            import cv2
            data = np.frombuffer(up.read(), np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        st.info("Capture or upload a frame — try two different faces and compare the stats.")
        return

    views = signals.blob_views(frame)
    st.write(f"Blob tensor shape: `{views.shape}`  (batch, channels, height, width)")

    r0, r1 = st.columns([1, 3])
    r0.image(bgr_to_rgb(views.resized_bgr), caption="Resized 300×300 (pre-subtraction)",
             use_container_width=True)
    r1.write("Per-channel statistics of the mean-subtracted signal "
             "(these numbers move as the face/lighting changes):")
    r1.table(_stats_table(views.stats))

    st.subheader("Mean-subtracted channel heatmaps")
    cols = st.columns(3)
    for col, hm, name in zip(cols, views.heatmaps, views.channel_names):
        col.image(bgr_to_rgb(hm), caption=name, use_container_width=True)

    st.subheader("Value distributions per channel")
    hcols = st.columns(3)
    for col, hist, stat, name in zip(hcols, views.histograms, views.stats, views.channel_names):
        edges = np.linspace(stat.min, stat.max, len(hist))
        col.caption(name)
        col.bar_chart({"value": edges, "count": hist}, x="value", y="count")
