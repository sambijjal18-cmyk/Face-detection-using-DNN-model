"""Face recognizers — identify WHO a detected face is.

Two recognizers, one per pipeline:

  LbphRecognizer       (pairs with Haar)  — histogram-based, cv2.face.LBPHFaceRecognizer
  EmbeddingRecognizer  (pairs with DNN)   — 128-D OpenFace vectors, cosine match

Both expose the same surface::

    fit(samples: list[(label, face_bgr)])  ->  trains from cropped faces
    save() / load()                        ->  persist to artifacts/
    identify(face_bgr) -> Prediction       ->  (name, score, is_known)

``score`` semantics differ (LBPH = distance, lower better; Embedding = cosine
distance, lower better) so callers should treat it as "lower is more confident".
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass(frozen=True)
class Prediction:
    name: str
    score: float  # lower = more confident, for both recognizers
    is_known: bool


def _valid(face: np.ndarray) -> bool:
    return face is not None and face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0


# --------------------------------------------------------------------------- #
# LBPH (Haar pipeline)
# --------------------------------------------------------------------------- #
class LbphRecognizer:
    name = "LBPH"

    def __init__(self) -> None:
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face is unavailable. Install opencv-contrib-python "
                "(not opencv-python): pip install -r requirements.txt"
            )
        self._model = cv2.face.LBPHFaceRecognizer_create()
        self._labels: dict[int, str] = {}
        self._trained = False
        self.threshold = config.LBPH_MATCH_THRESHOLD

    @staticmethod
    def _prep(face_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, config.LBPH_FACE_SIZE)

    def fit(self, samples: list[tuple[str, np.ndarray]]) -> None:
        name_to_id: dict[str, int] = {}
        images: list[np.ndarray] = []
        ids: list[int] = []
        for label, face in samples:
            if not _valid(face):
                continue
            label_id = name_to_id.setdefault(label, len(name_to_id))
            images.append(self._prep(face))
            ids.append(label_id)
        if not images:
            raise ValueError("LBPH.fit: no valid face samples")
        self._model.train(images, np.array(ids))
        self._labels = {v: k for k, v in name_to_id.items()}
        self._trained = True

    def identify(self, face_bgr: np.ndarray) -> Prediction:
        if not self._trained or not _valid(face_bgr):
            return Prediction(config.UNKNOWN_LABEL, float("inf"), False)
        label_id, distance = self._model.predict(self._prep(face_bgr))
        if distance > self.threshold:
            return Prediction(config.UNKNOWN_LABEL, float(distance), False)
        return Prediction(self._labels.get(label_id, config.UNKNOWN_LABEL), float(distance), True)

    def save(self) -> None:
        config.ensure_dirs()
        self._model.write(str(config.LBPH_MODEL))
        config.LABEL_MAP.write_text(json.dumps(self._labels))

    def load(self) -> bool:
        if not config.LBPH_MODEL.exists() or not config.LABEL_MAP.exists():
            return False
        self._model.read(str(config.LBPH_MODEL))
        raw = json.loads(config.LABEL_MAP.read_text())
        self._labels = {int(k): v for k, v in raw.items()}
        self._trained = True
        return True


# --------------------------------------------------------------------------- #
# OpenFace embeddings (DNN pipeline)
# --------------------------------------------------------------------------- #
class EmbeddingRecognizer:
    name = "Embeddings"

    def __init__(self) -> None:
        if not config.EMBEDDER_MODEL.exists():
            raise FileNotFoundError(
                "OpenFace embedder missing. Run: python scripts/download_models.py\n"
                f"  expected: {config.EMBEDDER_MODEL}"
            )
        self._net = cv2.dnn.readNetFromTorch(str(config.EMBEDDER_MODEL))
        self._names: list[str] = []
        self._centroids: np.ndarray | None = None  # (N, 128) L2-normalized mean per person
        self.threshold = config.EMBED_MATCH_THRESHOLD

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Return the L2-normalized 128-D embedding of one face crop."""
        blob = cv2.dnn.blobFromImage(
            face_bgr,
            config.EMBED_SCALE,
            config.EMBED_INPUT_SIZE,
            config.EMBED_MEAN,
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        vec = self._net.forward().flatten()
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def fit(self, samples: list[tuple[str, np.ndarray]]) -> None:
        per_person: dict[str, list[np.ndarray]] = {}
        for label, face in samples:
            if not _valid(face):
                continue
            per_person.setdefault(label, []).append(self.embed(face))
        if not per_person:
            raise ValueError("EmbeddingRecognizer.fit: no valid face samples")
        names: list[str] = []
        centroids: list[np.ndarray] = []
        for name, vecs in per_person.items():
            mean = np.mean(vecs, axis=0)
            mean /= np.linalg.norm(mean) or 1.0
            names.append(name)
            centroids.append(mean)
        self._names = names
        self._centroids = np.vstack(centroids)

    def identify(self, face_bgr: np.ndarray) -> Prediction:
        if self._centroids is None or not _valid(face_bgr):
            return Prediction(config.UNKNOWN_LABEL, float("inf"), False)
        vec = self.embed(face_bgr)
        # cosine distance = 1 - cosine similarity (both already L2-normalized)
        distances = 1.0 - self._centroids @ vec
        best = int(np.argmin(distances))
        dist = float(distances[best])
        if dist > self.threshold:
            return Prediction(config.UNKNOWN_LABEL, dist, False)
        return Prediction(self._names[best], dist, True)

    def save(self) -> None:
        config.ensure_dirs()
        if self._centroids is None:
            raise ValueError("nothing to save; call fit() first")
        np.savez(config.EMBEDDINGS_DB, names=np.array(self._names), centroids=self._centroids)

    def load(self) -> bool:
        if not config.EMBEDDINGS_DB.exists():
            return False
        data = np.load(config.EMBEDDINGS_DB, allow_pickle=True)
        self._names = list(data["names"])
        self._centroids = data["centroids"]
        return True
