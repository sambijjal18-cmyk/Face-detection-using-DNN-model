"""Compare the two pipelines on accuracy, detection robustness, and speed.

  Haar + LBPH        vs        DNN-SSD + Embeddings

Accuracy is measured with a held-out split of the *enrolled* faces (train on a
subset, test on the rest). With no independent labeled test set this is only
*indicative*, not a benchmark — the UI says so plainly. Detection robustness is
the mean number of faces found per dataset image. Speed is mean detect+identify
latency over the sampled faces, reported as FPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detectors import DnnSsdDetector, HaarDetector
from .enroll import load_dataset
from .recognizers import EmbeddingRecognizer, LbphRecognizer

# Monotonic clock injected so the module stays testable and avoids the banned
# Date.now()-style calls; defaults to time.perf_counter.
import time

_clock = time.perf_counter


@dataclass
class PipelineMetrics:
    pipeline: str
    accuracy: float          # 0..1 on the held-out split
    robustness: float        # mean faces detected per image
    fps: float               # mean frames/sec for detect+identify
    samples_tested: int
    notes: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    metrics: list[PipelineMetrics]
    verdict: dict[str, str]  # criterion -> winning pipeline
    overall: str
    indicative: bool = True


def _split(samples, holdout=0.3):
    """Per-person train/test split so every test person is also in training."""
    by_person: dict[str, list] = {}
    for name, face in samples:
        by_person.setdefault(name, []).append(face)
    train, test = [], []
    for name, faces in by_person.items():
        n_test = max(1, int(len(faces) * holdout)) if len(faces) > 1 else 0
        for i, face in enumerate(faces):
            (test if i < n_test else train).append((name, face))
    return train, test


def _eval_pipeline(name, detector, recognizer, train, test) -> PipelineMetrics:
    recognizer.fit(train)

    # Accuracy on held-out crops (recognition only; faces already cropped).
    correct = 0
    for true_name, face in test:
        if recognizer.identify(face).name == true_name:
            correct += 1
    accuracy = correct / len(test) if test else 0.0

    # Robustness + speed over the full dataset images (detect then identify).
    all_faces = train + test
    total_detected = 0
    t0 = _clock()
    for _, img in all_faces:
        dets = detector.detect(img)
        total_detected += len(dets)
        for d in dets:
            recognizer.identify(d.crop(img))
    elapsed = _clock() - t0

    robustness = total_detected / len(all_faces) if all_faces else 0.0
    fps = len(all_faces) / elapsed if elapsed > 0 else 0.0
    notes = []
    if not test:
        notes.append("too few samples per person for a held-out test; accuracy omitted")
    return PipelineMetrics(name, accuracy, robustness, fps, len(test), notes)


def run_comparison() -> ComparisonResult:
    """Train, evaluate, and rank both pipelines. Requires an enrolled dataset."""
    samples = load_dataset()
    if not samples:
        raise ValueError("dataset is empty — enroll people before comparing")
    train, test = _split(samples)

    haar = _eval_pipeline(
        "Haar + LBPH", HaarDetector(), LbphRecognizer(), train, test
    )
    dnn = _eval_pipeline(
        "DNN-SSD + Embeddings", DnnSsdDetector(), EmbeddingRecognizer(), train, test
    )

    def winner(key) -> str:
        return haar.pipeline if getattr(haar, key) >= getattr(dnn, key) else dnn.pipeline

    verdict = {
        "Recognition accuracy": winner("accuracy"),
        "Detection robustness": winner("robustness"),
        "Speed (FPS)": winner("fps"),
    }
    # Overall = majority vote across the three confirmed criteria.
    tally: dict[str, int] = {}
    for w in verdict.values():
        tally[w] = tally.get(w, 0) + 1
    overall = max(tally, key=tally.get)

    return ComparisonResult([haar, dnn], verdict, overall)
