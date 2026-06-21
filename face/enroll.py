"""Enrollment + training.

A person's reference faces live in ``dataset/<name>/*.jpg``. Two ways to populate
that folder:

  capture_from_webcam(name)        — grab N detected-and-cropped faces live
  add_images_from_folder(name, src)— copy existing images into dataset/<name>/

``load_dataset()`` reads every enrolled face back out, and ``train_all()`` rebuilds
both recognizers from it and persists them to ``artifacts/``.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detectors import DnnSsdDetector, HaarDetector
from .recognizers import EmbeddingRecognizer, LbphRecognizer

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _person_dir(name: str) -> Path:
    safe = "".join(c for c in name.strip() if c.isalnum() or c in (" ", "_", "-")).strip()
    if not safe:
        raise ValueError("enrollment name is empty after sanitizing")
    d = config.DATASET_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_people() -> dict[str, int]:
    """Map enrolled name -> number of stored images."""
    if not config.DATASET_DIR.exists():
        return {}
    out: dict[str, int] = {}
    for d in sorted(config.DATASET_DIR.iterdir()):
        if d.is_dir():
            out[d.name] = sum(1 for f in d.iterdir() if f.suffix.lower() in _IMG_EXTS)
    return out


def _largest_face(frame: np.ndarray, detector: HaarDetector | DnnSsdDetector) -> np.ndarray | None:
    dets = detector.detect(frame)
    if not dets:
        return None
    biggest = max(dets, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))
    crop = biggest.crop(frame)
    return crop if crop.size else None


def capture_from_webcam(
    name: str,
    samples: int = config.ENROLL_SAMPLES,
    cam_index: int = 0,
    on_sample=None,
) -> int:
    """Capture ``samples`` cropped faces from the webcam into dataset/<name>/.

    ``on_sample(i, total, preview_bgr)`` is called per saved sample so a UI can
    show progress. Returns the number of samples actually saved.
    """
    detector = DnnSsdDetector()  # better crops than Haar for enrollment
    dest = _person_dir(name)
    existing = sum(1 for f in dest.iterdir() if f.suffix.lower() in _IMG_EXTS)
    cap = cv2.VideoCapture(cam_index)
    saved = 0
    try:
        while saved < samples:
            ret, frame = cap.read()
            if not ret:
                break
            face = _largest_face(frame, detector)
            if face is not None:
                idx = existing + saved
                cv2.imwrite(str(dest / f"{idx:04d}.jpg"), face)
                saved += 1
                if on_sample is not None:
                    on_sample(saved, samples, face)
                time.sleep(0.1)  # spread samples across slightly different poses
    finally:
        cap.release()
    return saved


def add_images_from_folder(name: str, src_dir: str | Path) -> int:
    """Copy face images from ``src_dir`` into dataset/<name>/. Returns count copied."""
    src = Path(src_dir)
    if not src.is_dir():
        raise NotADirectoryError(src)
    dest = _person_dir(name)
    existing = sum(1 for f in dest.iterdir() if f.suffix.lower() in _IMG_EXTS)
    copied = 0
    for f in sorted(src.iterdir()):
        if f.suffix.lower() in _IMG_EXTS:
            shutil.copy(f, dest / f"{existing + copied:04d}{f.suffix.lower()}")
            copied += 1
    return copied


def load_dataset() -> list[tuple[str, np.ndarray]]:
    """Read all enrolled images as (name, face_bgr) pairs."""
    samples: list[tuple[str, np.ndarray]] = []
    if not config.DATASET_DIR.exists():
        return samples
    for person in sorted(config.DATASET_DIR.iterdir()):
        if not person.is_dir():
            continue
        for f in sorted(person.iterdir()):
            if f.suffix.lower() in _IMG_EXTS:
                img = cv2.imread(str(f))
                if img is not None and img.size:
                    samples.append((person.name, img))
    return samples


def train_all() -> dict[str, int]:
    """Rebuild and persist both recognizers from the dataset.

    Returns a small summary: number of people and total samples used.
    """
    samples = load_dataset()
    if not samples:
        raise ValueError("dataset is empty — enroll someone first")

    lbph = LbphRecognizer()
    lbph.fit(samples)
    lbph.save()

    emb = EmbeddingRecognizer()
    emb.fit(samples)
    emb.save()

    return {"people": len({n for n, _ in samples}), "samples": len(samples)}
