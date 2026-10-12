"""Load QC pass/fail thresholds from Tamil calibration JSON or config defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline import config


@dataclass
class QCThresholds:
    lse_d_max: float = 12.0
    lse_c_min: float = 1.0
    offset_abs_max: float = 5.0
    lse_d_z_max: float = 2.0
    lse_c_z_min: float = -2.0
    csim_min: float = 0.45
    csim_late_min: float = 0.40
    jitter_max: float = 0.35
    warping_max: float = 0.12
    seam_max: float = 0.25
    oral_artifact_max: float = 0.35
    bilabial_closure_fail_max: float = 0.40
    expr_drift_max: float = 0.15
    av_timing_ratio_min: float = 0.55
    av_timing_ratio_max: float = 1.85


_CACHE: Optional[QCThresholds] = None


def load_qc_thresholds(force: bool = False) -> QCThresholds:
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE
    t = QCThresholds(
        lse_d_max=float(getattr(config, "QC_LSE_D_MAX", 12.0)),
        lse_c_min=float(getattr(config, "QC_LSE_C_MIN", 1.0)),
        offset_abs_max=float(getattr(config, "QC_OFFSET_ABS_MAX", 5.0)),
        csim_min=float(getattr(config, "QC_CSIM_MIN", 0.45)),
        csim_late_min=float(getattr(config, "QC_CSIM_LATE_MIN", 0.40)),
        jitter_max=float(getattr(config, "QC_JITTER_MAX", 0.35)),
        warping_max=float(getattr(config, "QC_WARPING_MAX", 0.12)),
        seam_max=float(getattr(config, "QC_SEAM_MAX", 0.25)),
        oral_artifact_max=float(getattr(config, "QC_ORAL_MAX", 0.35)),
        bilabial_closure_fail_max=float(getattr(config, "QC_BILABIAL_FAIL_MAX", 0.40)),
        expr_drift_max=float(getattr(config, "QC_EXPR_DRIFT_MAX", 0.15)),
    )
    path = Path(config.TAMIL_CALIBRATION_PATH)
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as f:
                cal = json.load(f)
            if "qc_lse_d_max" in cal:
                t.lse_d_max = float(cal["qc_lse_d_max"])
            elif "healthy_lse_d_max" in cal:
                t.lse_d_max = float(cal["healthy_lse_d_max"])
            if "qc_lse_c_min" in cal:
                t.lse_c_min = float(cal["qc_lse_c_min"])
            elif "healthy_lse_c_min" in cal:
                t.lse_c_min = float(cal["healthy_lse_c_min"])
            if "qc_offset_abs_max" in cal:
                t.offset_abs_max = float(cal["qc_offset_abs_max"])
            else:
                off_std = float(cal.get("offset_std", 3.0))
                t.offset_abs_max = max(5.0, abs(float(cal.get("offset_median", 0))) + 2 * off_std)
            if cal.get("qc_csim_min") is not None:
                t.csim_min = float(cal["qc_csim_min"])
            if cal.get("av_timing_ratio_min") is not None:
                t.av_timing_ratio_min = float(cal["av_timing_ratio_min"])
            if cal.get("av_timing_ratio_max") is not None:
                t.av_timing_ratio_max = float(cal["av_timing_ratio_max"])
        except Exception:
            pass
    _CACHE = t
    return t
