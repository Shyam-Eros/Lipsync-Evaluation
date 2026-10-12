"""
Pipeline configuration — reference-free dubbed-movie evaluation.

CRITICAL: Never compare generated frames to English-synced source / GT video
frames as a correctness reference. Metrics are self-consistency or general
real-face distribution based only.

Edit TODO path placeholders, then: python run_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# TODO: set input dirs (Tamil-dubbed model outputs)
INPUT_DIRS = {
    "xdub": PROJECT_ROOT / "input_videos" / "xdub",
    "tbdub": PROJECT_ROOT / "input_videos" / "tbdub",
    "lipforcing": PROJECT_ROOT / "input_videos" / "lipforcing",
}

# Optional explicit pairs (filenames may differ). No GT field.
# EVAL_ITEMS = [{"video_id": "clip_3", "xdub": Path(...), "tbdub": Path(...), "lipforcing": Path(...)}]
EVAL_ITEMS = [
    {
        "video_id": "sample_2_opt",
        "xdub": Path("/local-workspace/gowtham.s/lip-sync_inference/x-dub/X-Dub/2_opt/job_eccd1d01_final.mp4"),
        "tbdub": Path("/local-workspace/gowtham.s/lip-sync_inference/TB-DUB/TBDub/results_1/video_tbdub.mp4"),
        "lipforcing": Path("/local-workspace/gowtham.s/lip-sync_inference/lip_forcing/LipForcing/out_sample3.mp4"),
    },
]

RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "logs"
TEMP_DIR = PROJECT_ROOT / "tmp"
QC_CLIPS_DIR = RESULTS_DIR / "qc_clips"

# Weights
SYNCNET_WEIGHTS_PATH = PROJECT_ROOT / "syncnet" / "data" / "syncnet_v2.model"
S3FD_WEIGHTS_PATH = (
    PROJECT_ROOT / "syncnet" / "detectors" / "s3fd" / "weights" / "sfd_face.pth"
)
# InsightFace model pack name (auto-download on first use) or local root
INSIGHTFACE_MODEL = "buffalo_l"
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface"

# Optional curated real-face crops for FID (never English-synced mouth GT)
REAL_FACE_REF_DIR = None  # TODO: Path(...) or leave None → use non_speaking_face crops

# Mouth / lower-face FID reference (built by calibrate_syncnet_tamil.py or set manually)
REAL_MOUTH_REF_DIR = None  # Path to directory of real Tamil mouth-crop JPGs

# Tamil SyncNet calibration (run calibrate_syncnet_tamil.py first)
TAMIL_CALIB_DATA_ROOT = None  # Path to your real Tamil talking-face clip datasets
TAMIL_CALIBRATION_PATH = RESULTS_DIR / "tamil_syncnet_calibration.json"

# I3D FVD weights (downloaded on first use)
I3D_WEIGHTS_PATH = PROJECT_ROOT / "models" / "i3d_torchscript.pt"
I3D_HF_REPO = "ByteDance/LatentSync-1.5"
I3D_HF_FILENAME = "auxiliary/i3d_torchscript.pt"

# Whisper (bilabial viseme checks)
WHISPER_PATH = Path(
    "/local-workspace/gowtham.s/lip-sync_inference/x-dub/X-Dub/checkpoints/whisper/large-v2.pt"
)

# Optional manual difficulty tags: video_id,tags
DIFFICULTY_MANIFEST_CSV = None  # Path(...) e.g. beard,occlusion,mic

# Layer 1 tag heuristics
LOW_LIGHT_LUMA_MAX = 70.0
FAST_MOTION_STD_MIN = 25.0
FAST_MOTION_SCALE_STD_MIN = 18.0
HIGH_SPEECH_RATE_MIN = 8.0
BILABIAL_MAR_MAX = 0.22
AV_TIMING_RATIO_MIN = 0.55
AV_TIMING_RATIO_MAX = 1.85

# QC defaults (overridden by tamil_syncnet_calibration.json when present)
QC_LSE_D_MAX = 12.0
QC_LSE_C_MIN = 1.0
QC_OFFSET_ABS_MAX = 5.0
QC_CSIM_MIN = 0.45
QC_CSIM_LATE_MIN = 0.40
QC_JITTER_MAX = 0.35
QC_WARPING_MAX = 0.12
QC_SEAM_MAX = 0.25
QC_ORAL_MAX = 0.35
QC_BILABIAL_FAIL_MAX = 0.40
QC_EXPR_DRIFT_MAX = 0.15

# ---------------------------------------------------------------------------
# Layer 1 thresholds
# ---------------------------------------------------------------------------
SHOT_DETECTION_SENSITIVITY = 27.0
FRAME_EXTRACTION_FPS = 25
AUDIO_SAMPLE_RATE = 16000
FACE_DETECT_CONF = 0.9
FACE_DETECT_SCALE = 0.25
FACE_CROP_SIZE = 224
FACE_CROP_SCALE = 0.40
MIN_TRACK_FRAMES = 25  # shorter than SyncNet-eval min; speaking segments filtered later
NUM_FAILED_DET = 25
MIN_FACE_SIZE = 40
IOU_TRACK_THRESH = 0.5
IDENTITY_SIM_THRESH = 0.45  # ArcFace cosine similarity for same identity
IDENTITY_TOP_N_CROPS = 5
ACTIVE_SPEAKER_MARGIN = 0.15  # LSE-C gap between #1 and #2
MIN_SEGMENT_DURATION = 0.4  # seconds
VAD_FRAME_MS = 30
VAD_AGGRESSIVENESS = 2  # webrtcvad 0-3
VAD_ENERGY_FALLBACK_DB = -40.0

# Shot scale by mean face area / frame area
SHOT_SCALE_CLOSEUP = 0.12
SHOT_SCALE_MEDIUM = 0.04  # else wide

# ---------------------------------------------------------------------------
# Layer 2 / SyncNet
# ---------------------------------------------------------------------------
SYNCNET_OFFSET_RANGE = 15
SYNCNET_BATCH_SIZE = 20
FID_BATCH_SIZE = 32
FID_DIMS = 2048
FVD_NUM_FRAMES = 16
FVD_MIN_FRAMES = 16
FVD_CLIP_SIZE = 224
MOUTH_LOWER_FRAC = 0.45  # bottom fraction of face crop used as lower-face ROI
MOUTH_OUT_SIZE = 128
CSIM_LATE_FRAC = 0.25  # last fraction of frames for late-clip identity
JITTER_LANDMARK_SUBSET = "lips"  # lips | face

# QC
QC_SAMPLE_RATE = 0.1
QC_CLOSE_UP_WEIGHT = 2.5
QC_SAMPLE_SEED = 42

MODEL_KEYS = ("xdub", "tbdub", "lipforcing")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Output CSV paths
SHOTS_CSV = RESULTS_DIR / "shots.csv"
TRACKLETS_CSV = RESULTS_DIR / "tracklets.csv"
SEGMENTS_CSV = RESULTS_DIR / "segments.csv"
METRICS_RAW_CSV = RESULTS_DIR / "metrics_raw.csv"
METRICS_BY_SCALE_CSV = RESULTS_DIR / "metrics_by_shot_scale.csv"
METRICS_BY_DIFFICULTY_CSV = RESULTS_DIR / "metrics_by_difficulty.csv"
METRICS_TIER_SUMMARY_JSON = RESULTS_DIR / "metrics_tier_summary.json"
QC_FLAGS_CSV = RESULTS_DIR / "qc_flags.csv"
QC_SHOTS_CSV = RESULTS_DIR / "qc_shots.csv"
QC_SAMPLE_CSV = RESULTS_DIR / "qc_sample.csv"
HUMAN_EVAL_DIR = RESULTS_DIR / "human_eval"
