"""Tab 1 — Live detect + recognize, both pipelines side by side."""

from __future__ import annotations

import time

import streamlit as st

from face import config
from face.detectors import draw_detections

from .common import (
    Camera,
    bgr_to_rgb,
    get_dnn,
    get_embedder,
    get_haar,
    get_lbph,
    missing_models_warning,
    recognizers_trained,
)

HAAR_COLOR = (255, 0, 0)   # blue (BGR)
DNN_COLOR = (0, 255, 0)    # green (BGR)


def _labels(detections, recognizer, frame):
    if recognizer is None:
        return [f"{d.confidence * 100:.0f}%" for d in detections]
    out = []
    for d in detections:
        pred = recognizer.identify(d.crop(frame))
        out.append(pred.name if pred.is_known else config.UNKNOWN_LABEL)
    return out


def render() -> None:
    st.header("Live — Detect + Recognize")
    st.caption("Haar + LBPH (blue) vs DNN-SSD + Embeddings (green), running on the same frames.")
    if missing_models_warning():
        return
    if not recognizers_trained():
        st.info("No one is enrolled yet — boxes will show but names will read **Unknown**. "
                "Go to the **Enroll** tab to register faces.")

    cam_index = st.number_input("Camera index", 0, 8, 0, key="live_cam")
    run = st.checkbox("Run camera", key="live_run")
    col_h, col_d = st.columns(2)
    haar_slot = col_h.empty()
    dnn_slot = col_d.empty()
    col_h.caption("Haar + LBPH")
    col_d.caption("DNN-SSD + Embeddings")
    fps_slot = st.empty()

    if not run:
        st.stop()

    haar, dnn = get_haar(), get_dnn()
    lbph, embedder = get_lbph(), get_embedder()
    embedder_for_id = embedder if recognizers_trained() else None

    with Camera(int(cam_index)) as cam:
        while st.session_state.get("live_run"):
            frame = cam.read()
            if frame is None:
                st.error("Could not read from camera.")
                break
            t0 = time.perf_counter()

            h_dets = haar.detect(frame)
            h_img = draw_detections(frame, h_dets, HAAR_COLOR, _labels(h_dets, lbph, frame))

            d_dets = dnn.detect(frame)
            d_img = draw_detections(frame, d_dets, DNN_COLOR, _labels(d_dets, embedder_for_id, frame))

            haar_slot.image(bgr_to_rgb(h_img), use_container_width=True)
            dnn_slot.image(bgr_to_rgb(d_img), use_container_width=True)
            dt = time.perf_counter() - t0
            fps_slot.metric("End-to-end FPS (both pipelines)", f"{1.0 / dt:.1f}" if dt else "—")
