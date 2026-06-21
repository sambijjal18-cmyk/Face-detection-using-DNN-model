"""Tab 5 — Compare the two pipelines and show which is best."""

from __future__ import annotations

import streamlit as st

from face import compare

from .common import missing_models_warning, recognizers_trained


def render() -> None:
    st.header("Compare — which pipeline is best?")
    st.caption("Judged on recognition accuracy, detection robustness, and speed (FPS).")
    if missing_models_warning():
        return
    if not recognizers_trained():
        st.info("Enroll at least one person (ideally a few, with several images each) in the "
                "**Enroll** tab first, then run the comparison.")
        return

    if not st.button("Run comparison", key="cmp_run"):
        return

    with st.spinner("Training + evaluating both pipelines on the enrolled dataset…"):
        try:
            result = compare.run_comparison()
        except ValueError as exc:
            st.error(str(exc))
            return

    st.subheader("Metrics")
    st.table([
        {
            "pipeline": m.pipeline,
            "accuracy": f"{m.accuracy * 100:.0f}%",
            "robustness (faces/img)": f"{m.robustness:.2f}",
            "FPS": f"{m.fps:.1f}",
            "test samples": m.samples_tested,
        }
        for m in result.metrics
    ])

    st.subheader("Winner by criterion")
    st.table([{"criterion": k, "winner": v} for k, v in result.verdict.items()])

    st.success(f"**Overall best: {result.overall}** (majority across the three criteria).")
    if result.indicative:
        st.caption("⚠️ Accuracy is measured on a held-out split of the *enrolled* faces, so it is "
                   "indicative — not a benchmark on independent data.")

    notes = [n for m in result.metrics for n in m.notes]
    if notes:
        st.caption("Notes: " + "; ".join(dict.fromkeys(notes)))
