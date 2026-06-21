"""Tab 2 — Signal Lab: how the RGB camera signal becomes grayscale (Haar path)."""

from __future__ import annotations

import numpy as np
import streamlit as st

from face import signals

from .common import Camera, bgr_to_rgb


def _hist_df(channel):
    return {"intensity": np.arange(256), "count": signals.histogram(channel)}


def render() -> None:
    st.header("Signal Lab — RGB → Grayscale")
    st.caption("The Haar detector works on a single grayscale signal. This is how the "
               "three color signals collapse into it.")
    st.latex(r"\text{Gray} = 0.299\,R + 0.587\,G + 0.114\,B")

    src = st.radio("Source", ["Snapshot from webcam", "Upload image"], horizontal=True, key="sig_src")
    frame = None
    if src == "Snapshot from webcam":
        cam_index = st.number_input("Camera index", 0, 8, 0, key="sig_cam")
        if st.button("Capture frame", key="sig_capture"):
            with Camera(int(cam_index)) as cam:
                frame = cam.read()
            if frame is not None:
                st.session_state["sig_frame"] = frame
        frame = st.session_state.get("sig_frame")
    else:
        up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp"], key="sig_upload")
        if up is not None:
            data = np.frombuffer(up.read(), np.uint8)
            import cv2
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        st.info("Capture or upload a frame to see its signals.")
        return

    views = signals.rgb_gray_views(frame)

    st.subheader("Channels")
    c0, c1, c2, c3 = st.columns(4)
    c0.image(bgr_to_rgb(frame), caption="Original (RGB)", use_container_width=True)
    c1.image(views.r, caption="R signal", use_container_width=True, clamp=True)
    c2.image(views.g, caption="G signal", use_container_width=True, clamp=True)
    c3.image(views.b, caption="B signal", use_container_width=True, clamp=True)

    st.subheader("Grayscale (weighted luma)")
    g0, g1 = st.columns([1, 2])
    g0.image(views.gray, caption="Grayscale", use_container_width=True, clamp=True)
    g1.write("Per-channel weights")
    g1.json({"R": views.weights[0], "G": views.weights[1], "B": views.weights[2]})

    st.subheader("Intensity histograms (the signals' distributions)")
    h0, h1, h2, h3 = st.columns(4)
    h0.caption("R"); h0.bar_chart(_hist_df(views.r), x="intensity", y="count")
    h1.caption("G"); h1.bar_chart(_hist_df(views.g), x="intensity", y="count")
    h2.caption("B"); h2.bar_chart(_hist_df(views.b), x="intensity", y="count")
    h3.caption("Gray"); h3.bar_chart(_hist_df(views.gray), x="intensity", y="count")
