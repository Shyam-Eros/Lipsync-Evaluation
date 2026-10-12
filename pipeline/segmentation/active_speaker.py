"""Layer 1 stage 4 — active speaker via VAD + SyncNet LSE-C; emit segments.csv."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile

from pipeline import config
from pipeline.io_utils import append_csv, slice_audio, write_csv
from pipeline.metrics.shared_face_crop import list_crop_paths

logger = logging.getLogger(__name__)

SEGMENT_FIELDS = [
    "segment_id",
    "segment_type",
    "video_id",
    "model_name",
    "shot_id",
    "tracklet_id",
    "identity_id",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "duration_s",
    "shot_scale",
    "audio_slice_path",
    "face_crop_dir",
    "exclude_reason",
    "notes",
    "difficulty_tags",
]


def _energy_vad(audio: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """Fallback energy VAD → list of (start_s, end_s) speech intervals."""
    frame = int(sr * config.VAD_FRAME_MS / 1000)
    if frame < 1:
        frame = 480
    thresh = 10 ** (config.VAD_ENERGY_FALLBACK_DB / 20.0) * 32768.0
    if audio.dtype != np.int16:
        audio_i = (np.clip(audio, -1, 1) * 32767).astype(np.int16) if audio.dtype != np.int16 else audio
    else:
        audio_i = audio
    speech = []
    for i in range(0, len(audio_i) - frame, frame):
        chunk = audio_i[i : i + frame].astype(np.float32)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        speech.append(rms >= thresh)
    # Merge
    intervals = []
    in_sp = False
    start = 0
    for i, flag in enumerate(speech):
        t = i * frame / sr
        if flag and not in_sp:
            in_sp = True
            start = t
        elif not flag and in_sp:
            in_sp = False
            intervals.append((start, t))
    if in_sp:
        intervals.append((start, len(audio_i) / sr))
    return intervals


def detect_speech_intervals(wav_path: Path) -> List[Tuple[float, float]]:
    sr, audio = wavfile.read(str(wav_path))
    try:
        import webrtcvad

        vad = webrtcvad.Vad(int(config.VAD_AGGRESSIVENESS))
        if audio.dtype != np.int16:
            audio = (np.clip(audio.astype(np.float32), -1, 1) * 32767).astype(np.int16)
        frame = int(sr * config.VAD_FRAME_MS / 1000)
        # webrtcvad requires 8000/16000/32000/48000 and specific frame sizes
        if sr not in (8000, 16000, 32000, 48000):
            return _energy_vad(audio, sr)
        bytes_per = frame * 2
        speech_flags = []
        raw = audio.tobytes()
        for i in range(0, len(raw) - bytes_per, bytes_per):
            try:
                speech_flags.append(vad.is_speech(raw[i : i + bytes_per], sr))
            except Exception:
                speech_flags.append(False)
        intervals = []
        in_sp = False
        start = 0.0
        for i, flag in enumerate(speech_flags):
            t = i * frame / sr
            if flag and not in_sp:
                in_sp, start = True, t
            elif not flag and in_sp:
                in_sp = False
                intervals.append((start, t))
        if in_sp:
            intervals.append((start, len(audio) / sr))
        return intervals if intervals else _energy_vad(audio, sr)
    except Exception as exc:
        logger.warning("webrtcvad unavailable (%s); energy VAD", exc)
        return _energy_vad(audio, sr)


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def run_active_speaker(
    *,
    video_id: str,
    model_name: str,
    shots: List[Dict],
    tracklets: List[Dict],
    identity_map: Dict[int, int],
    shot_scales: Dict[int, str],
    full_wav: Path,
    work_dir: Path,
    fps: float | None = None,
    score_tracklet_fn=None,
) -> List[Dict]:
    """
    Build final segments.csv rows.

    score_tracklet_fn(crop_dir, audio_wav) -> (lse_d, lse_c) optional; if None,
    uses SyncNet from metrics.sync_eval when available, else energy-only ASD.
    """
    fps = float(fps if fps is not None else config.FRAME_EXTRACTION_FPS)
    work_dir = Path(work_dir)
    speech_iv = detect_speech_intervals(full_wav)
    by_shot: Dict[int, List[Dict]] = {}
    for t in tracklets:
        by_shot.setdefault(int(t["shot_id"]), []).append(t)

    segments: List[Dict] = []
    seg_id = 0

    if score_tracklet_fn is None:
        try:
            from pipeline.metrics.sync_eval import evaluate_sync_track

            def score_tracklet_fn(crop_dir, audio_wav):
                d, c, _ = evaluate_sync_track(crop_dir, audio_wav)
                return d, c
        except Exception:
            score_tracklet_fn = None

    for shot in shots:
        sid = int(shot["shot_id"])
        scale = shot_scales.get(sid, "pending")
        s0 = float(shot["start_time"])
        s1 = float(shot["end_time"])
        tracks = by_shot.get(sid, [])

        if not tracks:
            row = {
                "segment_id": f"s{seg_id}",
                "segment_type": "no_face",
                "video_id": video_id,
                "model_name": model_name,
                "shot_id": sid,
                "tracklet_id": "",
                "identity_id": "",
                "start_frame": shot["start_frame"],
                "end_frame": shot["end_frame"],
                "start_time": shot["start_time"],
                "end_time": shot["end_time"],
                "duration_s": f"{s1 - s0:.4f}",
                "shot_scale": scale,
                "audio_slice_path": "",
                "face_crop_dir": "",
                "exclude_reason": "no_face",
                "notes": "valid_no_visible_face",
            }
            segments.append(row)
            seg_id += 1
            continue

        # Speech overlap with this shot
        shot_speech = [
            (max(s0, a), min(s1, b)) for a, b in speech_iv if _overlap(s0, s1, a, b) > 0.05
        ]

        if not shot_speech or score_tracklet_fn is None:
            # All tracklets = non_speaking_face
            for t in tracks:
                tid = int(t["tracklet_id"])
                row = {
                    "segment_id": f"s{seg_id}",
                    "segment_type": "non_speaking_face",
                    "video_id": video_id,
                    "model_name": model_name,
                    "shot_id": sid,
                    "tracklet_id": tid,
                    "identity_id": identity_map.get(tid, ""),
                    "start_frame": t["start_frame"],
                    "end_frame": t["end_frame"],
                    "start_time": f"{int(t['start_frame']) / fps:.4f}",
                    "end_time": f"{(int(t['end_frame']) + 1) / fps:.4f}",
                    "duration_s": f"{(int(t['end_frame']) - int(t['start_frame']) + 1) / fps:.4f}",
                    "shot_scale": scale,
                    "audio_slice_path": "",
                    "face_crop_dir": t["crop_dir"],
                    "exclude_reason": "non_speaking" if shot_speech else "no_speech_vad",
                    "notes": "face_present_no_active_speaker",
                }
                segments.append(row)
                seg_id += 1
            continue

        # Score each tracklet on shot-level speech audio (merged)
        sp0 = min(a for a, _ in shot_speech)
        sp1 = max(b for _, b in shot_speech)
        shot_wav = work_dir / f"shot{sid:03d}_speech.wav"
        slice_audio(full_wav, shot_wav, sp0, sp1)

        scores = []
        for t in tracks:
            crop_dir = Path(t["crop_dir"])
            if len(list_crop_paths(crop_dir)) < 6:
                continue
            try:
                lse_d, lse_c = score_tracklet_fn(crop_dir, shot_wav)
                scores.append((t, lse_d, lse_c))
            except Exception as exc:
                logger.warning("ASD SyncNet failed tracklet %s: %s", t["tracklet_id"], exc)

        if not scores:
            for t in tracks:
                tid = int(t["tracklet_id"])
                row = {
                    "segment_id": f"s{seg_id}",
                    "segment_type": "non_speaking_face",
                    "video_id": video_id,
                    "model_name": model_name,
                    "shot_id": sid,
                    "tracklet_id": tid,
                    "identity_id": identity_map.get(tid, ""),
                    "start_frame": t["start_frame"],
                    "end_frame": t["end_frame"],
                    "start_time": f"{int(t['start_frame']) / fps:.4f}",
                    "end_time": f"{(int(t['end_frame']) + 1) / fps:.4f}",
                    "duration_s": f"{(int(t['end_frame']) - int(t['start_frame']) + 1) / fps:.4f}",
                    "shot_scale": scale,
                    "audio_slice_path": "",
                    "face_crop_dir": t["crop_dir"],
                    "exclude_reason": "asd_score_failed",
                    "notes": "",
                }
                segments.append(row)
                seg_id += 1
            continue

        scores.sort(key=lambda x: -x[2])
        best_t, best_d, best_c = scores[0]
        second_c = scores[1][2] if len(scores) > 1 else -1e9
        margin = best_c - second_c

        for t, lse_d, lse_c in scores:
            tid = int(t["tracklet_id"])
            is_active = (
                t is best_t and margin >= config.ACTIVE_SPEAKER_MARGIN
            ) or (t is best_t and len(scores) == 1)

            t0 = int(t["start_frame"]) / fps
            t1 = (int(t["end_frame"]) + 1) / fps
            # Intersect with speech
            seg_start = max(t0, sp0)
            seg_end = min(t1, sp1)
            dur = seg_end - seg_start

            if is_active and dur >= config.MIN_SEGMENT_DURATION:
                audio_path = work_dir / f"seg{seg_id:04d}_audio.wav"
                slice_audio(full_wav, audio_path, seg_start, seg_end)
                # Subcrop dir: reuse full tracklet crops (Layer 2 uses audio slice)
                row = {
                    "segment_id": f"s{seg_id}",
                    "segment_type": "speaking",
                    "video_id": video_id,
                    "model_name": model_name,
                    "shot_id": sid,
                    "tracklet_id": tid,
                    "identity_id": identity_map.get(tid, ""),
                    "start_frame": int(seg_start * fps),
                    "end_frame": int(seg_end * fps),
                    "start_time": f"{seg_start:.4f}",
                    "end_time": f"{seg_end:.4f}",
                    "duration_s": f"{dur:.4f}",
                    "shot_scale": scale,
                    "audio_slice_path": str(audio_path),
                    "face_crop_dir": t["crop_dir"],
                    "exclude_reason": "",
                    "notes": f"active_lse_c={lse_c:.4f};margin={margin:.4f}",
                }
            elif is_active and dur < config.MIN_SEGMENT_DURATION:
                row = {
                    "segment_id": f"s{seg_id}",
                    "segment_type": "non_speaking_face",
                    "video_id": video_id,
                    "model_name": model_name,
                    "shot_id": sid,
                    "tracklet_id": tid,
                    "identity_id": identity_map.get(tid, ""),
                    "start_frame": t["start_frame"],
                    "end_frame": t["end_frame"],
                    "start_time": f"{t0:.4f}",
                    "end_time": f"{t1:.4f}",
                    "duration_s": f"{t1 - t0:.4f}",
                    "shot_scale": scale,
                    "audio_slice_path": "",
                    "face_crop_dir": t["crop_dir"],
                    "exclude_reason": "below_min_segment_duration",
                    "notes": f"dur={dur:.3f}",
                }
            else:
                row = {
                    "segment_id": f"s{seg_id}",
                    "segment_type": "non_speaking_face",
                    "video_id": video_id,
                    "model_name": model_name,
                    "shot_id": sid,
                    "tracklet_id": tid,
                    "identity_id": identity_map.get(tid, ""),
                    "start_frame": t["start_frame"],
                    "end_frame": t["end_frame"],
                    "start_time": f"{t0:.4f}",
                    "end_time": f"{t1:.4f}",
                    "duration_s": f"{t1 - t0:.4f}",
                    "shot_scale": scale,
                    "audio_slice_path": "",
                    "face_crop_dir": t["crop_dir"],
                    "exclude_reason": "non_speaking",
                    "notes": f"lse_c={lse_c:.4f}",
                }
            segments.append(row)
            seg_id += 1

    n_speak = sum(1 for r in segments if r["segment_type"] == "speaking")
    print(
        f"{model_name}/{video_id} — active speaker: {n_speak} speaking segments "
        f"({len(segments)} total rows)",
        flush=True,
    )

    frames_dir = work_dir.parent / "frames"
    if frames_dir.is_dir() and tracklets:
        from pipeline.segmentation.difficulty_tags import enrich_segments

        segments = enrich_segments(segments, tracklets, frames_dir, full_wav, video_id)
    else:
        for s in segments:
            s.setdefault("difficulty_tags", "")

    # Write / append segments for this video
    write_mode_append = config.SEGMENTS_CSV.is_file() and config.SEGMENTS_CSV.stat().st_size > 0
    if write_mode_append:
        append_csv(config.SEGMENTS_CSV, segments, SEGMENT_FIELDS)
    else:
        write_csv(config.SEGMENTS_CSV, segments, SEGMENT_FIELDS)
    return segments
