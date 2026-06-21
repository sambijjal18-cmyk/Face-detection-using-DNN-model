"""Download the three model files this project needs into ``models/``.

  - deploy.prototxt                          (SSD face detector architecture)
  - res10_300x300_ssd_iter_140000.caffemodel (SSD face detector weights)
  - openface_nn4.small2.v1.t7                (128-D face embedder)

Run once after install::

    python scripts/download_models.py

If a download URL ever rots, the file can be fetched manually from the URLs
printed below and dropped into ``models/`` by hand.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Make the `face` package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face import config  # noqa: E402

# (destination path, source URL)
FILES: list[tuple[Path, str]] = [
    (
        config.SSD_PROTOTXT,
        "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt",
    ),
    (
        config.SSD_WEIGHTS,
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    ),
    (
        config.EMBEDDER_MODEL,
        "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7",
    ),
]


def _download(dest: Path, url: str) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {dest.name} already present")
        return
    print(f"[get ] {dest.name}\n        from {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)  # noqa: S310 (trusted URLs)
    except Exception as exc:  # surface a clear, actionable message
        print(f"[FAIL] {dest.name}: {exc}\n        Download manually from the URL above into {dest.parent}")
        return
    print(f"[ok  ] {dest.name} ({dest.stat().st_size // 1024} KB)")


def main() -> None:
    config.ensure_dirs()
    for dest, url in FILES:
        _download(dest, url)
    print("\nDone. Model files live in:", config.MODELS_DIR)


if __name__ == "__main__":
    main()
