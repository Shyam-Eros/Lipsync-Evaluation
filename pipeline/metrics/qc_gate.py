"""Per-segment QC pass/fail vs calibrated thresholds."""

from __future__ import annotations

from typing import Dict, List, Optional

from pipeline.qc_thresholds import load_qc_thresholds


def _f(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def evaluate_row(row: dict) -> dict:
    """Return qc_pass, qc_fail_reasons for one metrics_raw row."""
    out = {"qc_pass": "", "qc_fail_reasons": ""}
    if row.get("segment_type") != "speaking":
        out["qc_pass"] = ""
        out["qc_fail_reasons"] = "skipped_non_speaking"
        return out
    t = load_qc_thresholds()
    reasons: List[str] = []

    lse_d = _f(row.get("lse_d"))
    lse_c = _f(row.get("lse_c"))
    off = _f(row.get("lse_offset_corrected")) or _f(row.get("lse_offset"))
    lse_d_z = _f(row.get("lse_d_z"))
    lse_c_z = _f(row.get("lse_c_z"))

    if lse_d is not None and lse_d > t.lse_d_max:
        reasons.append(f"lse_d>{t.lse_d_max:.2f}")
    if lse_c is not None and lse_c < t.lse_c_min:
        reasons.append(f"lse_c<{t.lse_c_min:.2f}")
    if off is not None and abs(off) > t.offset_abs_max:
        reasons.append(f"offset>{t.offset_abs_max:.1f}")
    if lse_d_z is not None and lse_d_z > t.lse_d_z_max:
        reasons.append(f"lse_d_z>{t.lse_d_z_max:.2f}")
    if lse_c_z is not None and lse_c_z < t.lse_c_z_min:
        reasons.append(f"lse_c_z<{t.lse_c_z_min:.2f}")

    csim = _f(row.get("csim"))
    if csim is not None and csim < t.csim_min:
        reasons.append(f"csim<{t.csim_min:.2f}")
    csim_late = _f(row.get("csim_late"))
    if csim_late is not None and csim_late < t.csim_late_min:
        reasons.append(f"csim_late<{t.csim_late_min:.2f}")

    jit = _f(row.get("jitter_score"))
    if jit is not None and jit > t.jitter_max:
        reasons.append(f"jitter>{t.jitter_max:.3f}")
    warp = _f(row.get("warping_error"))
    if warp is not None and warp > t.warping_max:
        reasons.append(f"warp>{t.warping_max:.3f}")
    seam = _f(row.get("seam_score"))
    if seam is not None and seam > t.seam_max:
        reasons.append(f"seam>{t.seam_max:.3f}")
    oral = _f(row.get("oral_artifact_score"))
    if oral is not None and oral > t.oral_artifact_max:
        reasons.append(f"oral>{t.oral_artifact_max:.3f}")
    bfail = _f(row.get("bilabial_closure_fail_rate"))
    if bfail is not None and bfail > t.bilabial_closure_fail_max:
        reasons.append(f"bilabial_fail>{t.bilabial_closure_fail_max:.2f}")
    drift = _f(row.get("expr_drift"))
    if drift is not None and drift > t.expr_drift_max:
        reasons.append(f"expr_drift>{t.expr_drift_max:.3f}")
    ratio = _f(row.get("av_timing_ratio"))
    if ratio is not None and (ratio < t.av_timing_ratio_min or ratio > t.av_timing_ratio_max):
        reasons.append("timing_stress")

    notes = str(row.get("notes", ""))
    if row.get("fvd_mode") == "proxy" or "fvd_proxy" in notes:
        reasons.append("fvd_proxy_used")

    out["qc_pass"] = "0" if reasons else "1"
    out["qc_fail_reasons"] = ";".join(reasons)
    return out


def apply_qc_flags(rows: List[dict]) -> List[dict]:
    flagged = []
    for r in rows:
        qc = evaluate_row(r)
        nr = dict(r)
        nr.update(qc)
        flagged.append(nr)
    return flagged


def rollup_qc_shots(rows: List[dict]) -> List[dict]:
    """Worst segment per (video_id, model_name, shot_id)."""
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        if r.get("segment_type") != "speaking":
            continue
        key = (r.get("video_id"), r.get("model_name"), r.get("shot_id"))
        groups.setdefault(key, []).append(r)
    out = []
    for (vid, model, shot), segs in sorted(groups.items()):
        fails = [s for s in segs if s.get("qc_pass") == "0"]
        out.append(
            {
                "video_id": vid,
                "model_name": model,
                "shot_id": shot,
                "n_speaking_segments": len(segs),
                "n_fail": len(fails),
                "shot_qc_pass": "0" if fails else "1",
                "fail_reasons": ";".join(
                    sorted({s.get("qc_fail_reasons", "") for s in fails if s.get("qc_fail_reasons")})
                ),
            }
        )
    return out
