"""Face detectors behind a uniform interface.

Both ``HaarDetector`` and ``DnnSsdDetector`` expose::

    detect(frame_bgr) -> list[Detection]

where each ``Detection`` is a clamped (x1, y1, x2, y2, confidence) box in pixel
coordinates. Haar has no real confidence score, so it reports 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Return the face sub-image (may be empty if the box is degenerate)."""
        return frame[self.y1 : self.y2, self.x1 : self.x2]


def _clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return x1, y1, x2, y2


class HaarDetector:
    """Classic Viola–Jones cascade. Operates on the grayscale signal."""

    name = "Haar"

    def __init__(self) -> None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(xml)
        if self._cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {xml}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        out: list[Detection] = []
        for (x, y, fw, fh) in faces:
            x1, y1, x2, y2 = _clamp_box(int(x), int(y), int(x + fw), int(y + fh), w, h)
            if x2 > x1 and y2 > y1:
                out.append(Detection(x1, y1, x2, y2, 1.0))
        return out


class DnnSsdDetector:
    """res10 SSD Caffe model. Operates on a 300x300 mean-subtracted blob."""

    name = "DNN-SSD"

    def __init__(self) -> None:
        if not config.SSD_PROTOTXT.exists() or not config.SSD_WEIGHTS.exists():
            raise FileNotFoundError(
                "SSD model files missing. Run: python scripts/download_models.py\n"
                f"  expected: {config.SSD_PROTOTXT}\n            {config.SSD_WEIGHTS}"
            )
        self._net = cv2.dnn.readNetFromCaffe(
            str(config.SSD_PROTOTXT), str(config.SSD_WEIGHTS)
        )
        self.conf_threshold = config.SSD_CONF_THRESHOLD

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, config.SSD_INPUT_SIZE),
            1.0,
            config.SSD_INPUT_SIZE,
            config.SSD_MEAN,
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        out: list[Detection] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.conf_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = _clamp_box(*box.astype("int"), w, h)
            if x2 > x1 and y2 > y1:
                out.append(Detection(x1, y1, x2, y2, confidence))
        return out


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    color: tuple[int, int, int],
    labels: list[str] | None = None,
) -> np.ndarray:
    """Draw boxes (+ optional per-box labels) on a copy of ``frame``."""
    out = frame.copy()
    for idx, det in enumerate(detections):
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        text = labels[idx] if labels and idx < len(labels) else f"{det.confidence * 100:.0f}%"
        cv2.putText(
            out, text, (det.x1, max(15, det.y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return out
