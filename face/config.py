"""Central configuration: paths, model locations, thresholds.

Everything is anchored to the project root so paths work on any machine — this
replaces the machine-specific ``C:\\Users\\sambi\\...`` paths the original DNN
script hardcoded.
"""

from __future__ import annotations

from pathlib import Path

# Project root = parent of this `face/` package.
ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT / "models"
DATASET_DIR = ROOT / "dataset"
ARTIFACTS_DIR = ROOT / "artifacts"

# --- Model files (downloaded by scripts/download_models.py) -------------------
SSD_PROTOTXT = MODELS_DIR / "deploy.prototxt"
SSD_WEIGHTS = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
EMBEDDER_MODEL = MODELS_DIR / "openface_nn4.small2.v1.t7"

# --- Trained recognizer artifacts ---------------------------------------------
LBPH_MODEL = ARTIFACTS_DIR / "lbph.yml"
LABEL_MAP = ARTIFACTS_DIR / "label_map.json"
EMBEDDINGS_DB = ARTIFACTS_DIR / "embeddings.npz"

# --- DNN-SSD detector ----------------------------------------------------------
SSD_INPUT_SIZE = (300, 300)
SSD_MEAN = (104.0, 177.0, 123.0)  # BGR mean subtracted inside blobFromImage
SSD_CONF_THRESHOLD = 0.5

# --- OpenFace embedder ---------------------------------------------------------
EMBED_INPUT_SIZE = (96, 96)
EMBED_MEAN = (0.0, 0.0, 0.0)
EMBED_SCALE = 1.0 / 255.0
# Max cosine distance to still count as a match; above this -> "Unknown".
EMBED_MATCH_THRESHOLD = 0.45

# --- LBPH recognizer -----------------------------------------------------------
LBPH_FACE_SIZE = (200, 200)  # grayscale crops are resized to this before training
# LBPH returns a *distance* (lower = more confident). Above this -> "Unknown".
LBPH_MATCH_THRESHOLD = 70.0

# --- Enrollment ---------------------------------------------------------------
ENROLL_SAMPLES = 20  # webcam samples captured per person by default

UNKNOWN_LABEL = "Unknown"


def ensure_dirs() -> None:
    """Create the runtime directories if they don't exist."""
    for d in (MODELS_DIR, DATASET_DIR, ARTIFACTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
