"""Shared I/O helpers (ffmpeg extract, CSV append) — not a second segmenter."""

from __future__ import annotations

import csv
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pipeline import config


def ensure_dirs() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    config.QC_CLIPS_DIR.mkdir(parents=True, exist_ok=True)


def cache_key(video_path: Path) -> str:
    return hashlib.sha1(str(Path(video_path).resolve()).encode()).hexdigest()[:16]


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")


def extract_audio(video_path: Path, wav_out: Path, sample_rate: int | None = None) -> Path:
    ensure_ffmpeg()
    sample_rate = sample_rate or config.AUDIO_SAMPLE_RATE
    wav_out = Path(wav_out)
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-async", "1", "-ac", "1", "-vn",
        "-acodec", "pcm_s16le", "-ar", str(sample_rate),
        str(wav_out),
    ]
    subprocess.run(cmd, check=True)
    return wav_out


def extract_frames(video_path: Path, out_dir: Path, fps: float | None = None) -> List[Path]:
    ensure_ffmpeg()
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = fps if fps is not None else config.FRAME_EXTRACTION_FPS
    pattern = str(out_dir / "%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-threads", "1", "-f", "image2", pattern,
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(out_dir.glob("*.jpg"))
    if not frames:
        raise RuntimeError(f"No frames from {video_path}")
    return frames


def slice_audio(
    full_wav: Path,
    out_wav: Path,
    start_s: float,
    end_s: float,
) -> Path:
    ensure_ffmpeg()
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(full_wav),
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-ac", "1", "-acodec", "pcm_s16le",
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def export_clip(
    video_path: Path,
    out_path: Path,
    start_s: float,
    end_s: float,
) -> Path:
    ensure_ffmpeg()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", str(video_path),
        "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path


def append_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def read_csv(path: Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
