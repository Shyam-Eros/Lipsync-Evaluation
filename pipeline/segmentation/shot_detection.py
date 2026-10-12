"""Layer 1 stage 1 — content-aware shot / scene cut detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from pipeline import config
from pipeline.io_utils import append_csv

logger = logging.getLogger(__name__)

SHOT_FIELDS = [
    "shot_id",
    "video_id",
    "model_name",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "shot_scale",
    "n_frames",
]


def detect_shots_from_video(
    video_path: Path,
    *,
    threshold: float | None = None,
) -> List[Tuple[int, int]]:
    """Return (start_frame, end_frame_exclusive) in source video frame indices."""
    from scenedetect import ContentDetector, SceneManager, open_video

    threshold = float(
        threshold if threshold is not None else config.SHOT_DETECTION_SENSITIVITY
    )
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video)
    scene_list = manager.get_scene_list()
    if not scene_list:
        return []
    return [(s.get_frames(), e.get_frames()) for s, e in scene_list]


def detect_shots_from_frames(
    frames_dir: Path,
    *,
    threshold: float | None = None,
) -> List[Tuple[int, int]]:
    """Histogram-based fallback on extracted JPG sequence."""
    import cv2

    flist = sorted(Path(frames_dir).glob("*.jpg"))
    n = len(flist)
    if n == 0:
        return []
    threshold = float(
        threshold if threshold is not None else config.SHOT_DETECTION_SENSITIVITY
    )
    cut_score = max(0.35, min(0.95, threshold / 60.0))
    prev = None
    cuts = [0]
    for i, path in enumerate(flist):
        img = cv2.imread(str(path))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev is not None:
            dist = cv2.compareHist(prev, hist, cv2.HISTCMP_BHATTACHARYYA)
            if dist >= cut_score:
                cuts.append(i)
        prev = hist
    cuts.append(n)
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]


def map_shots_to_extracted_fps(
    shots: List[Tuple[int, int]],
    n_extracted: int,
) -> List[Tuple[int, int]]:
    if not shots:
        return [(0, n_extracted)]
    last = max(b for _, b in shots) or 1
    scale = n_extracted / float(last)
    mapped = [
        (max(0, int(a * scale)), min(n_extracted, int(b * scale))) for a, b in shots
    ]
    mapped = [(a, b) for a, b in mapped if b > a]
    return mapped or [(0, n_extracted)]


def run_shot_detection(
    video_path: Path,
    frames_dir: Path,
    *,
    video_id: str,
    model_name: str,
    fps: float | None = None,
    write_csv: bool = True,
) -> List[Dict]:
    """
    Detect shots and optionally append to results/shots.csv incrementally.
    shot_scale left as 'pending' until faces are known.
    """
    fps = float(fps if fps is not None else config.FRAME_EXTRACTION_FPS)
    n_frames = len(sorted(Path(frames_dir).glob("*.jpg")))
    try:
        raw = detect_shots_from_video(video_path)
        shots = map_shots_to_extracted_fps(raw, n_frames) if raw else detect_shots_from_frames(frames_dir)
    except Exception as exc:
        logger.warning("PySceneDetect failed (%s); using frame histogram", exc)
        shots = detect_shots_from_frames(frames_dir)

    if not shots:
        shots = [(0, n_frames)]

    rows: List[Dict] = []
    for sid, (a, b) in enumerate(shots):
        row = {
            "shot_id": sid,
            "video_id": video_id,
            "model_name": model_name,
            "start_frame": a,
            "end_frame": b,
            "start_time": f"{a / fps:.4f}",
            "end_time": f"{b / fps:.4f}",
            "shot_scale": "pending",
            "n_frames": b - a,
        }
        rows.append(row)
        if write_csv:
            append_csv(config.SHOTS_CSV, [row], SHOT_FIELDS)

    logger.info(
        "%s/%s — shot detection: %d shots found", model_name, video_id, len(rows)
    )
    print(
        f"{model_name}/{video_id} — shot detection: {len(rows)} shots found",
        flush=True,
    )
    return rows
