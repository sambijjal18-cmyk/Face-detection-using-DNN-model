"""Tab 4 — Enroll people (webcam capture and/or image folder) and train."""

from __future__ import annotations

import streamlit as st

from face import config, enroll

from .common import bgr_to_rgb, clear_recognizer_cache, missing_models_warning


def render() -> None:
    st.header("Enroll & Train")
    if missing_models_warning():
        return

    people = enroll.list_people()
    if people:
        st.write("Currently enrolled:")
        st.table([{"name": n, "images": c} for n, c in people.items()])
    else:
        st.info("No one enrolled yet.")

    st.subheader("1 · Add a person")
    tab_cam, tab_folder = st.tabs(["From webcam", "From image folder"])

    with tab_cam:
        name = st.text_input("Name", key="enr_cam_name")
        samples = st.slider("Samples to capture", 5, 60, config.ENROLL_SAMPLES, key="enr_samples")
        cam_index = st.number_input("Camera index", 0, 8, 0, key="enr_cam_idx")
        if st.button("Capture from webcam", key="enr_cam_btn", disabled=not name.strip()):
            progress = st.progress(0.0)
            preview = st.empty()

            def on_sample(i, total, face):
                progress.progress(i / total)
                preview.image(bgr_to_rgb(face), caption=f"sample {i}/{total}", width=160)

            saved = enroll.capture_from_webcam(name, int(samples), int(cam_index), on_sample)
            st.success(f"Saved {saved} samples for {name!r}.")

    with tab_folder:
        fname = st.text_input("Name", key="enr_dir_name")
        src = st.text_input("Folder path containing this person's images", key="enr_dir_path")
        if st.button("Import folder", key="enr_dir_btn", disabled=not (fname.strip() and src.strip())):
            try:
                copied = enroll.add_images_from_folder(fname, src)
                st.success(f"Imported {copied} images for {fname!r}.")
            except (NotADirectoryError, ValueError) as exc:
                st.error(str(exc))

    st.subheader("2 · Train recognizers")
    st.caption("Rebuilds both LBPH and the embedding database from everyone enrolled.")
    if st.button("Train now", key="enr_train"):
        try:
            summary = enroll.train_all()
            clear_recognizer_cache()
            st.success(f"Trained on {summary['samples']} samples across {summary['people']} people. "
                       "Recognition is now live in the other tabs.")
        except ValueError as exc:
            st.error(str(exc))
