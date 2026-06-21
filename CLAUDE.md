# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A learning/PBL project comparing two face pipelines — **Haar cascade + LBPH** vs **DNN-SSD + OpenFace embeddings** — for real-time detection AND recognition, with a Streamlit UI that also visualizes the signal-processing internals (RGB→grayscale, and the SSD input blob). Everything stays in the OpenCV family; no dlib/face_recognition.

Two layers coexist:

1. **Streamlit app** (`app.py` + `face/` + `ui/`) — the main deliverable.
2. **Legacy single-file demos** (kept as-is) — `Face_Detection_Project(PBL).py` (Haar, detect-only, blue boxes) and `#Face detection using DNN model.py` (DNN-SSD, detect-only, green boxes). These still run standalone but are superseded by the app.

## Running

```
pip install -r requirements.txt          # IMPORTANT: opencv-contrib-python, not opencv-python
python scripts/download_models.py         # fetch the 3 model files into models/
streamlit run app.py                      # open the 5-tab app
```

Legacy demos still work directly: `python "Face_Detection_Project(PBL).py"`.

There is no test suite, lint config, or build step. Verify changes by smoke-testing modules
(`python -c "from face... import ..."`) and booting the app headless
(`streamlit run app.py --server.headless true --server.port <port>` → expect HTTP 200).

## Why opencv-contrib (non-obvious)

The LBPH recognizer is `cv2.face.LBPHFaceRecognizer_create()`, and `cv2.face` ships **only** in
`opencv-contrib-python`. Plain `opencv-python` will import fine but `cv2.face` is absent and recognition
breaks. `requirements.txt` pins the contrib build for this reason.

## Architecture

- **`face/config.py`** — single source of paths/thresholds, all anchored to project root via `pathlib`
  (this replaced the old hardcoded `C:\Users\sambi\...` model paths). `ensure_dirs()` creates runtime dirs.
- **`face/detectors.py`** — `HaarDetector` and `DnnSsdDetector` behind a uniform
  `detect(frame) -> list[Detection]` (clamped x1,y1,x2,y2,confidence). `draw_detections()` for overlay.
- **`face/signals.py`** — the teaching core (pure, no camera/Streamlit): `rgb_gray_views()` (R/G/B + luma
  grayscale, exposes the `0.299/0.587/0.114` weights), `histogram()`, and `blob_views()` which unpacks the
  SSD's `blobFromImage` `(1,3,300,300)` tensor into 3 mean-subtracted channels + per-channel stats/heatmaps.
- **`face/recognizers.py`** — `LbphRecognizer` (Haar partner; trains on grayscale crops, distance score)
  and `EmbeddingRecognizer` (DNN partner; 128-D OpenFace vectors, cosine match). Both: `fit / save / load /
  identify`. `identify()` returns a `Prediction(name, score, is_known)` where **lower score = more
  confident** for both. Above the per-recognizer threshold → `Unknown`.
- **`face/enroll.py`** — populate `dataset/<name>/*.jpg` via `capture_from_webcam()` or
  `add_images_from_folder()`; `train_all()` rebuilds + persists both recognizers from the dataset.
- **`face/compare.py`** — `run_comparison()` does a per-person train/test split, scores both pipelines on
  accuracy / detection robustness / FPS, and picks an overall winner by majority vote. Accuracy is
  **indicative** (held-out split of enrolled faces, not an independent benchmark) — keep that caveat visible.
- **`ui/`** — one `render()` per tab: `live`, `signal_lab`, `blob_lab`, `enroll_ui`, `compare_ui`.
  `ui/common.py` holds shared helpers (BGR→RGB, `Camera` context manager, `@st.cache_resource` singletons).
  `app.py` is just the 5-tab router.

## Data/artifact locations (all gitignored)

- `models/` — downloaded weights: `deploy.prototxt`, `res10_300x300_ssd_iter_140000.caffemodel`,
  `openface_nn4.small2.v1.t7`.
- `dataset/<name>/*.jpg` — enrolled reference faces.
- `artifacts/` — trained recognizers: `lbph.yml`, `label_map.json`, `embeddings.npz`.

## Conventions / gotchas

- OpenCV is **BGR**; Streamlit `st.image` expects **RGB** — convert with `ui.common.bgr_to_rgb` before display.
- After (re)training, call `ui.common.clear_recognizer_cache()` so cached recognizers reload.
- Streamlit live capture uses **server-side `cv2.VideoCapture(0)`** (works for local single-machine use).
  For true in-browser webcam you'd add `streamlit-webrtc` — not currently a dependency.
- `face/signals.py` must stay pure (no Streamlit/camera imports) so it stays testable and reusable.
- The leading `#` in `#Face detection using DNN model.py` is part of the filename, not a comment — quote it.
- Legacy DNN demo still has the old hardcoded `C:\Users\sambi\...` paths; the app does **not** (uses `config`).
