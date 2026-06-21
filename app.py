"""Streamlit entry point — face detection + recognition + signal labs.

Run with:

    streamlit run app.py

Tabs:
  Live        — both pipelines detecting + recognizing on the same frames
  Signal Lab  — RGB → grayscale signal transformation (Haar path)
  BLOB Lab    — how the SSD input blob varies face-to-face (DNN path)
  Enroll      — register people (webcam / folder) and train recognizers
  Compare     — accuracy + robustness + FPS, and which pipeline wins
"""

from __future__ import annotations

import streamlit as st

from face import config
from ui import blob_lab, compare_ui, enroll_ui, live, signal_lab

st.set_page_config(page_title="Face Detect + Recognize", layout="wide")
config.ensure_dirs()

st.title("Face Detection + Recognition — Signals & Pipeline Comparison")

tabs = st.tabs(["Live", "Signal Lab (RGB→Gray)", "BLOB Lab (DNN)", "Enroll", "Compare"])

with tabs[0]:
    live.render()
with tabs[1]:
    signal_lab.render()
with tabs[2]:
    blob_lab.render()
with tabs[3]:
    enroll_ui.render()
with tabs[4]:
    compare_ui.render()
