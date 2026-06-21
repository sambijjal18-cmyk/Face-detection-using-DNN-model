"""Shared helpers for the Streamlit tabs: color conversion, camera, recognizers."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from face import config
from face.detectors import DnnSsdDetector, HaarDetector
from face.recognizers import EmbeddingRecognizer, LbphRecognizer


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """OpenCV is BGR; Streamlit/st.image expects RGB."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def models_present() -> bool:
    return config.SSD_PROTOTXT.exists() and config.SSD_WEIGHTS.exists() and config.EMBEDDER_MODEL.exists()


def missing_models_warning() -> bool:
    """Render a warning if model files are absent. Returns True if missing."""
    if models_present():
        return False
    st.warning(
        "Model files are missing. Run **`python scripts/download_models.py`** "
        "to fetch them into `models/`, then reload."
    )
    return True


# --- cached singletons (loaded once per session) ------------------------------
@st.cache_resource
def get_haar() -> HaarDetector:
    return HaarDetector()


@st.cache_resource
def get_dnn() -> DnnSsdDetector:
    return DnnSsdDetector()


@st.cache_resource
def get_lbph() -> LbphRecognizer | None:
    rec = LbphRecognizer()
    return rec if rec.load() else None


@st.cache_resource
def get_embedder() -> EmbeddingRecognizer | None:
    rec = EmbeddingRecognizer()
    rec_loaded = rec.load()
    # The Torch net is always loaded; centroids only if previously trained.
    return rec if rec_loaded else rec


def recognizers_trained() -> bool:
    return config.LBPH_MODEL.exists() and config.EMBEDDINGS_DB.exists()


def clear_recognizer_cache() -> None:
    """Call after (re)training so the cached recognizers reload fresh."""
    get_lbph.clear()
    get_embedder.clear()


class Camera:
    """Tiny context manager around cv2.VideoCapture for Streamlit run-loops."""

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        self.cap = cv2.VideoCapture(self.index)
        return self

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def __exit__(self, *exc) -> None:
        if self.cap is not None:
            self.cap.release()
