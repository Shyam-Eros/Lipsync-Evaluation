"""Bilabial closure check using Whisper word timestamps + mouth aspect ratio."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from pipeline import config
from pipeline.metrics.mouth_geometry import frame_index_for_time, mar_at_crop_index

logger = logging.getLogger(__name__)

_WHISPER_MODEL = None

# Tamil + Latin graphemes that often correspond to bilabial closures
_BILABIAL_CHARS = set("பப்பமம்ப")


def _get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL if _WHISPER_MODEL is not False else None
    try:
        import whisper

        path = getattr(config, "WHISPER_PATH", None)
        name = "large-v2"
        if path and Path(path).is_file():
            _WHISPER_MODEL = whisper.load_model(path)
        else:
            _WHISPER_MODEL = whisper.load_model(name)
        return _WHISPER_MODEL
    except Exception as exc:
        logger.warning("Whisper unavailable (%s)", exc)
        _WHISPER_MODEL = False
        return None


def _is_bilabial_token(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if any(c in _BILABIAL_CHARS for c in t):
        return True
    if re.match(r"^[pbmPB]+$", t):
        return True
    return False


def _bilabial_windows(
    audio_path: Path,
    seg_start: float,
    seg_end: float,
) -> List[Tuple[float, float]]:
    model = _get_whisper()
    if model is None:
        return []
    result = model.transcribe(
        str(audio_path),
        language="ta",
        word_timestamps=True,
        verbose=False,
    )
    windows = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            word = str(w.get("word", ""))
            if not _is_bilabial_token(word):
                continue
            ws = float(w.get("start", 0))
            we = float(w.get("end", ws + 0.05))
            if we < seg_start or ws > seg_end:
                continue
            windows.append((max(ws, seg_start), min(we, seg_end)))
    return windows


def evaluate_segment(segment: dict) -> dict:
    out = {
        "bilabial_frames": "",
        "bilabial_closure_fail_rate": "",
        "bilabial_score": "",
        "viseme_notes": "",
    }
    if segment.get("segment_type") != "speaking":
        out["viseme_notes"] = "skipped_non_speaking"
        return out
    audio = segment.get("audio_slice_path")
    crop = segment.get("face_crop_dir")
    if not audio or not crop:
        out["viseme_notes"] = "missing_paths"
        return out
    seg_start = float(segment.get("start_time", 0))
    seg_end = float(segment.get("end_time", seg_start))
    try:
        windows = _bilabial_windows(Path(audio), seg_start, seg_end)
        if not windows:
            out["viseme_notes"] = "no_bilabial_tokens"
            out["bilabial_score"] = "1.000000"
            return out
        fps = float(config.FRAME_EXTRACTION_FPS)
        mars = []
        fails = 0
        closure_thresh = float(getattr(config, "BILABIAL_MAR_MAX", 0.22))
        for ws, we in windows:
            t = (ws + we) / 2.0
            rel_t = t - seg_start
            idx = frame_index_for_time(Path(crop), rel_t, fps=fps)
            mar = mar_at_crop_index(Path(crop), idx)
            if mar is None:
                continue
            mars.append(mar)
            if mar > closure_thresh:
                fails += 1
        n = len(mars)
        if n == 0:
            out["viseme_notes"] = "no_mar_at_bilabial"
            return out
        fail_rate = fails / n
        out["bilabial_frames"] = str(n)
        out["bilabial_closure_fail_rate"] = f"{fail_rate:.6f}"
        out["bilabial_score"] = f"{max(0.0, 1.0 - fail_rate):.6f}"
    except Exception as exc:
        out["viseme_notes"] = f"viseme_failed:{exc}"
    return out
