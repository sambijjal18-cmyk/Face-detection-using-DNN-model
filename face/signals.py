"""Signal-level views of the two pipelines — the teaching core of the project.

Haar path  -> how the RGB camera signal collapses to a single grayscale signal.
DNN path   -> how the blobFromImage tensor (the SSD's actual input signal) shifts
              face-to-face after resize + mean subtraction.

Everything here is pure (no camera, no Streamlit) so it is trivially testable and
reusable by the UI tabs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config

# ITU-R BT.601 luma weights — what cv2.COLOR_BGR2GRAY actually applies.
LUMA_WEIGHTS = (0.299, 0.587, 0.114)  # (R, G, B)
LUMA_FORMULA = "Gray = 0.299*R + 0.587*G + 0.114*B"


# --------------------------------------------------------------------------- #
# RGB -> grayscale (Haar path)
# --------------------------------------------------------------------------- #
@dataclass
class RgbGrayViews:
    """Per-channel signals plus the luma grayscale, all uint8 single-channel."""

    r: np.ndarray
    g: np.ndarray
    b: np.ndarray
    gray: np.ndarray
    weights: tuple[float, float, float]
    formula: str


def rgb_gray_views(frame_bgr: np.ndarray) -> RgbGrayViews:
    """Split a BGR frame into R/G/B signals and the weighted luma grayscale."""
    b, g, r = cv2.split(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return RgbGrayViews(
        r=r, g=g, b=b, gray=gray, weights=LUMA_WEIGHTS, formula=LUMA_FORMULA
    )


def histogram(channel: np.ndarray, bins: int = 256) -> np.ndarray:
    """256-bin intensity histogram of a single-channel uint8 signal."""
    hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
    return hist.flatten().astype(np.int64)


def colorize(channel: np.ndarray, bgr_tint: tuple[int, int, int]) -> np.ndarray:
    """Map a single-channel signal onto a colored image for display (BGR)."""
    norm = channel.astype(np.float32) / 255.0
    tint = np.array(bgr_tint, dtype=np.float32)
    return (norm[..., None] * tint).astype(np.uint8)


# --------------------------------------------------------------------------- #
# blobFromImage (DNN path)
# --------------------------------------------------------------------------- #
@dataclass
class ChannelStats:
    name: str
    min: float
    max: float
    mean: float
    std: float


@dataclass
class BlobViews:
    """Unpacked SSD input blob for visualization.

    The SSD's real input is blobFromImage(...) of shape (1, 3, 300, 300): the
    frame resized to 300x300 with the BGR mean (104, 177, 123) subtracted. These
    mean-subtracted channels ARE the signal the network sees, and they shift with
    every different face / lighting condition.
    """

    resized_bgr: np.ndarray  # 300x300 BGR, pre-mean-subtraction (for reference)
    channels: list[np.ndarray]  # 3 x (300x300) float32, mean-subtracted (B,G,R order)
    channel_names: list[str]
    heatmaps: list[np.ndarray]  # 3 x (300x300x3) uint8 colormapped, for display
    histograms: list[np.ndarray]  # 3 x bins, over the mean-subtracted values
    stats: list[ChannelStats]
    shape: tuple[int, ...]  # the raw blob shape, e.g. (1, 3, 300, 300)


def blob_views(frame_bgr: np.ndarray, hist_bins: int = 64) -> BlobViews:
    """Compute the SSD input blob and unpack it for visualization."""
    resized = cv2.resize(frame_bgr, config.SSD_INPUT_SIZE)
    blob = cv2.dnn.blobFromImage(
        resized, 1.0, config.SSD_INPUT_SIZE, config.SSD_MEAN
    )  # shape (1, 3, H, W); channels are B, G, R after mean subtraction

    names = ["B - 104", "G - 177", "R - 123"]
    channels: list[np.ndarray] = []
    heatmaps: list[np.ndarray] = []
    hists: list[np.ndarray] = []
    stats: list[ChannelStats] = []

    for c in range(3):
        ch = blob[0, c, :, :].astype(np.float32)
        channels.append(ch)
        # Normalize to 0..255 for a stable colormap, then apply JET.
        norm = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmaps.append(cv2.applyColorMap(norm, cv2.COLORMAP_JET))
        hists.append(
            np.histogram(ch, bins=hist_bins, range=(ch.min(), ch.max()))[0].astype(np.int64)
        )
        stats.append(
            ChannelStats(
                name=names[c],
                min=float(ch.min()),
                max=float(ch.max()),
                mean=float(ch.mean()),
                std=float(ch.std()),
            )
        )

    return BlobViews(
        resized_bgr=resized,
        channels=channels,
        channel_names=names,
        heatmaps=heatmaps,
        histograms=hists,
        stats=stats,
        shape=tuple(blob.shape),
    )
