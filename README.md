# Lip-Sync Evaluation (Evalution)

Reference-free evaluation for **inferred** lip-sync videos (Tamil dub audio muxed in the file).

The pipeline does **not** compare outputs to English-synced source frames. That comparison is invalid for dubbing: the original lips match English, not Tamil.

| | |
|---|---|
| **Entry point** | `run_pipeline.py` |
| **Calibration** | `calibrate_syncnet_tamil.py` |
| **Project root** | `/local-workspace/gowtham.s/lip-sync_inference/Evalution` |
| **Deep dive** | [`workflow.md`](workflow.md) |

---

## Pipeline at a glance

```text
Inferred MP4 (Tamil audio in file)
  │
  ├─ Layer 1 — segmentation
  │     shot detect → face tracklets → identity clusters → active speaker
  │     → segments.csv (+ difficulty_tags per segment)
  │
  ├─ Layer 2 — metrics (per segment row)
  │     Tier 1: LSE (+ Tamil z-scores) | mouth FID | I3D FVD | CSIM (+ late) | jitter | warping
  │     Tier 2: seam | oral artifacts | Whisper bilabial | upper-face retention | AV timing
  │     → metrics_raw.csv
  │
  └─ Post-process
        QC gates → qc_flags.csv, qc_shots.csv
        Rollups → metrics_by_shot_scale.csv, metrics_by_difficulty.csv
        Summary → metrics_tier_summary.json
        Human pack → results/human_eval/ (clips + scorecards)
```

One `run_pipeline.py` invocation should include **every** video you want in the same comparison table (base + finetuned, all job IDs). Each full run **rewrites** aggregate CSVs under `results/`.

---

## Tier 1 metrics (automatic QC core)

Run **`calibrate_syncnet_tamil.py` before production eval** so LSE z-scores, I3D FVD, and QC thresholds are meaningful.

| Column | Better | Meaning |
|--------|--------|---------|
| `lse_d` | lower | SyncNet distance: mouth motion vs **this file’s** dubbed audio |
| `lse_c` | higher | SyncNet confidence |
| `lse_offset` | ~0 frames | Estimated AV lag |
| `lse_d_z`, `lse_c_z` | ~0 vs real Tamil | Z-scores from `tamil_syncnet_calibration.json` |
| `lse_offset_corrected` | ~0 | Offset minus calibration median bias |
| `fid` | lower | **Mouth / lower-face** FID (Inception on mouth ROI) |
| `fid_face` | lower | Optional full-face FID |
| `fvd` | lower | Mouth **motion** distance (I3D features vs Tamil real refs) |
| `fvd_mode` | `i3d` | `proxy` if I3D weights/refs unavailable — treat FVD as indicative only |
| `csim` | higher | ArcFace similarity vs best same-identity embedding **in this clip** |
| `csim_late` | higher | CSIM on last 25% of segment frames (long-clip drift) |
| `jitter_score` | lower | Temporal flicker (landmarks or optical flow) |
| `warping_error` | lower | Frame-to-frame instability in the face crop |

**Tier 1 checklist**

1. `bash syncnet/download_weights.sh`
2. `python calibrate_syncnet_tamil.py --data-root <tamil_clips>` → JSON + `tamil_mouth_ref/`
3. Confirm `results/tamil_syncnet_calibration.json` exists before trusting `qc_pass`
4. Compare models on the **same** `video_id` and slice by `shot_scale` (never one global mean)

---

## Tier 2 metrics (production QC extensions)

Computed on **speaking** segments unless noted. Failures are logged per segment; other branches still run.

| Column | Better | Meaning |
|--------|--------|---------|
| `seam_score` | lower | Laplacian energy at mouth-crop top/bottom bands vs interior (blend seam) |
| `oral_artifact_score` | lower | High-frequency / color heuristic in open-mouth oral ROI |
| `bilabial_frames` | — | Frames sampled on Whisper-timed bilabial intervals |
| `bilabial_closure_fail_rate` | lower | Share of bilabial frames where mouth aspect ratio (MAR) is not closed |
| `bilabial_score` | higher | `1 - fail_rate` (when bilabial tokens exist) |
| `upper_face_csim` | higher | Upper-face ArcFace vs same-identity non-speaking ref in clip |
| `expr_drift` | lower | Upper-face embedding variance across segment |
| `av_timing_ratio` | ~1 | Active speech (VAD) vs lip-motion active duration |
| `timing_stress_flag` | 0 preferred | 1 if ratio outside calibrated band |

**Whisper (bilabial):** set `WHISPER_PATH` in [`pipeline/config.py`](pipeline/config.py) to a local `large-v2.pt` (or install `openai-whisper` and allow download). First viseme pass can be slow.

**I3D FVD:** weights download on first use to `models/i3d_torchscript.pt` (see `I3D_HF_REPO` in config). Mouth reference dirs come from calibration (`tamil_mouth_ref/`) or `REAL_MOUTH_REF_DIR`.

---

## Layer 1: `difficulty_tags`

Comma-separated tags on each `segments.csv` row (auto heuristics + optional manifest).

| Tag | Source |
|-----|--------|
| `close_up`, `medium`, `wide` | From `shot_scale` |
| `low_light` | Mean face-crop luminance below threshold |
| `fast_motion` | High bbox center / scale variance |
| `profile` | Bbox aspect / yaw proxy |
| `high_speech_rate` | Energy-peak rate on segment audio |

Optional **`DIFFICULTY_MANIFEST_CSV`** in config: columns `video_id`, `tags` for manual labels (beard, occlusion, mic, etc.).

Rollup: [`results/metrics_by_difficulty.csv`](results/metrics_by_difficulty.csv) — means per **model × tag** (do not average across tags into one number).

---

## QC gates (`qc_pass`)

After Layer 2, [`pipeline/metrics/qc_gate.py`](pipeline/metrics/qc_gate.py) evaluates each **speaking** segment against thresholds from:

1. **`tamil_syncnet_calibration.json`** (`qc_lse_d_max`, `qc_lse_c_min`, `qc_offset_abs_max`, …) when present  
2. Else defaults in [`pipeline/config.py`](pipeline/config.py) (`QC_LSE_D_MAX`, …)

| Output | Content |
|--------|---------|
| `qc_pass` | `1` pass, `0` fail (empty for non-speaking) |
| `qc_fail_reasons` | Semicolon-separated rule hits (e.g. `lse_d>12.00`, `fvd_proxy_used`) |
| `qc_flags.csv` | Speaking segments only (subset of columns) |
| `qc_shots.csv` | Shot-level rollup: fail if any speaking segment in shot fails |

**Note:** Without calibration, `fvd_mode=proxy` often adds `fvd_proxy_used` to fail reasons. Calibrate and use I3D FVD for production gates.

---

## Human evaluation (MOS + A/B)

Automatic metrics do not replace native-speaker judgment. After each pipeline run:

| Path | Purpose |
|------|---------|
| `results/human_eval/clips/` | Short exported segments |
| `results/human_eval/mos_scorecard.csv` | 1–5 MOS: sync, naturalness, quality |
| `results/human_eval/ab_pairs.csv` | Paired `xdub_base` vs `xdub_ft` (same scene) |
| `results/human_eval/README_raters.md` | Instructions for Tamil raters |

Rebuild only the pack (if metrics already exist):

```bash
python build_human_eval_pack.py --metrics results/metrics_raw.csv --qc results/qc_flags.csv
```

After raters fill in `rater_id` and scores:

```bash
python aggregate_human_mos.py
# → results/human_eval/human_mos_summary.json
```

`results/qc_sample.csv` remains a lightweight stratified sample; prefer `human_eval/` for formal studies.

---

## Prerequisites

1. **ffmpeg** on `PATH`
2. Python venv at `Evalution/.venv`
3. **SyncNet + S3FD** weights (once)

```bash
cd /local-workspace/gowtham.s/lip-sync_inference/Evalution
source .venv/bin/activate
pip install -r requirements.txt
bash syncnet/download_weights.sh
```

**GPU (recommended):**

```bash
export CUDA_VISIBLE_DEVICES=0
```

**Smoke tests (no full video run):**

```bash
python -m unittest tests.test_metrics_smoke -q
```

---

## Input rules

- **Model outputs only** — `run_final.mp4`, inference export, etc.
- Audio in the file must be the **dubbed Tamil** track, not the original English sync.
- Do **not** use side-by-side compare videos or raw `input.mp4` as eval input.

**UI-UX Inference layout:**

```text
.../data/jobs/<job_id>/infer_base/run_final.mp4   → base model (e.g. xdub_base)
.../data/jobs/<job_id>/infer_ft/run_final.mp4     → finetuned (e.g. xdub_ft)
```

`video_id` defaults to `<job_id>` when the path contains `.../jobs/<job_id>/...`.

---

## Recommended workflow

### Step 1 — Tamil SyncNet calibration

```bash
python calibrate_syncnet_tamil.py \
  --data-root /path/to/your/tamil/talking-face/clips \
  --max-clips 50
```

**Writes:**

- `results/tamil_syncnet_calibration.json` — LSE stats, z-score params, **QC thresholds**, timing ratio bands
- `results/tamil_mouth_ref/` — real mouth crops for FID / FVD references

Set `TAMIL_CALIB_DATA_ROOT` in [`pipeline/config.py`](pipeline/config.py) to avoid repeating `--data-root`.

If you skip calibration: raw LSE still runs; `lse_*_z` stay empty; FVD tends toward **proxy**; QC is less trustworthy.

---

### Step 2 — Run evaluation

#### A) UI jobs script (two job IDs, four videos)

```bash
CUDA_VISIBLE_DEVICES=0 bash run_ui_jobs_eval.sh
```

Runs **one** `run_pipeline.py` with all base + finetuned paths. Edit `JOB_IDS` in [`run_ui_jobs_eval.sh`](run_ui_jobs_eval.sh) for other jobs.

#### B) CLI — ad-hoc videos

Same clip, two models:

```bash
python run_pipeline.py \
  --video /path/to/base_out.mp4 --model xdub_base \
  --video /path/to/ft_out.mp4 --model xdub_ft \
  --video-id my_clip_01
```

Multiple jobs in one run:

```bash
python run_pipeline.py \
  --video .../jobs/5e106411bce3/infer_base/run_final.mp4 --model xdub_base \
  --video .../jobs/5e106411bce3/infer_ft/run_final.mp4 --model xdub_ft \
  --video .../jobs/c9145455d460/infer_base/run_final.mp4 --model xdub_base \
  --video .../jobs/c9145455d460/infer_ft/run_final.mp4 --model xdub_ft
```

#### C) Config batch

Set `EVAL_ITEMS` or `INPUT_DIRS` in [`pipeline/config.py`](pipeline/config.py), then:

```bash
python run_pipeline.py
```

#### Useful flags

| Flag | Purpose |
|------|---------|
| `--layer1-only` | Segmentation only → `segments.csv` |
| `--reuse-segments` | Skip Layer 1 when segments already exist for these videos |

---

## Outputs

| File | Content |
|------|---------|
| [`results/segments.csv`](results/segments.csv) | Segment types, times, crops, `difficulty_tags` |
| [`results/metrics_raw.csv`](results/metrics_raw.csv) | Tier 1 + Tier 2 + QC columns per segment |
| [`results/metrics_by_shot_scale.csv`](results/metrics_by_shot_scale.csv) | Means by model × shot scale |
| [`results/metrics_by_difficulty.csv`](results/metrics_by_difficulty.csv) | Means by model × difficulty tag |
| [`results/qc_flags.csv`](results/qc_flags.csv) | Speaking-segment QC |
| [`results/qc_shots.csv`](results/qc_shots.csv) | Per-shot QC rollup |
| [`results/metrics_tier_summary.json`](results/metrics_tier_summary.json) | QC pass rate, Tier 2 means, rollups |
| [`results/human_eval/`](results/human_eval/) | Rater pack (auto-built at end of pipeline) |
| [`results/qc_sample.csv`](results/qc_sample.csv) | Small stratified clip sample |
| [`results/logs/run.log`](results/logs/run.log) | Main log |
| [`results/logs/<model>_<video_id>_metrics.log`](results/logs/) | Per-video metric branch errors |

The console prints a **SUMMARY** (Tier 1 table, Tier 2 means, QC pass %, FVD proxy warning).

---

## How to compare base vs finetuned

On the **same** `video_id`:

1. **Sync (primary):** lower `lse_d`, higher `lse_c`; check `lse_*_z` after calibration.
2. **Mouth appearance:** lower `fid`; confirm `fvd_mode` is `i3d` before trusting `fvd`.
3. **Identity:** higher `csim` / `csim_late`.
4. **Stability:** lower `jitter_score`, `warping_error`, `seam_score`.
5. **Tier 2:** lower `oral_artifact_score`, `bilabial_closure_fail_rate`, `expr_drift`; higher `bilabial_score`, `upper_face_csim`.
6. **Gates:** `% qc_pass` in summary and `metrics_tier_summary.json`.

Always slice by **`shot_scale`** and **`difficulty_tags`** — never report a single blended average across close-up and wide shots.

---

## Configuration reference

Edit [`pipeline/config.py`](pipeline/config.py) for paths and thresholds:

| Setting | Role |
|---------|------|
| `TAMIL_CALIBRATION_PATH` | Calibration + QC JSON |
| `REAL_MOUTH_REF_DIR` / calib mouth ref | FID/FVD real distribution |
| `WHISPER_PATH` | Local Whisper checkpoint for bilabial timing |
| `I3D_WEIGHTS_PATH`, `I3D_HF_*` | I3D FVD weights download |
| `DIFFICULTY_MANIFEST_CSV` | Manual difficulty tags per `video_id` |
| `QC_*` | Default QC limits if JSON missing |
| `EVAL_ITEMS` / `INPUT_DIRS` | Batch video list |

---

## Common warnings

| Message | Effect |
|---------|--------|
| No Tamil calibration JSON | No z-scores; FVD proxy; QC may flag `fvd_proxy_used` |
| I3D FVD failed | Falls back to temporal proxy; `fvd_mode=proxy` |
| InsightFace on CPU | Slower CSIM; results still valid |
| MediaPipe unavailable | Jitter / MAR fallbacks (optical flow, gradients) |
| Whisper missing / slow | Bilabial columns empty or first run downloads weights |
| webrtcvad missing | Energy-based VAD for active speaker |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No videos found | Pass `--video` / `--model` or set `EVAL_ITEMS` / `INPUT_DIRS` |
| Missing SyncNet weights | `bash syncnet/download_weights.sh` |
| Missing `infer_ft/run_final.mp4` | Finish inference; check job `status.json` and logs |
| Only last job in CSV | Use one combined `run_pipeline.py` or `run_ui_jobs_eval.sh` |
| All segments `qc_pass=0` | Run calibration; check `fvd_mode` and `qc_fail_reasons` |
| Human pack empty clips | Re-run with same `--video` paths so `jobs_index` resolves |

---

## Quick checklist

1. `bash syncnet/download_weights.sh`
2. `python calibrate_syncnet_tamil.py --data-root <tamil_clips>`
3. Set `WHISPER_PATH` (and GPU) if you need Tier 2 bilabial metrics
4. `CUDA_VISIBLE_DEVICES=0 bash run_ui_jobs_eval.sh` **or** `python run_pipeline.py --video ... --model ...`
5. Review `metrics_raw.csv`, `qc_flags.csv`, `metrics_by_shot_scale.csv`, `metrics_by_difficulty.csv`, console SUMMARY
6. Ship `results/human_eval/` to raters; then `python aggregate_human_mos.py`

---

## Related scripts

| Script | Purpose |
|--------|---------|
| `run_pipeline.py` | Layer 1 + 2 + QC + rollups + human pack |
| `calibrate_syncnet_tamil.py` | Tamil SyncNet + mouth refs + QC thresholds |
| `run_ui_jobs_eval.sh` | Preset UI job batch eval |
| `build_human_eval_pack.py` | Regenerate MOS/A/B pack from existing metrics |
| `aggregate_human_mos.py` | Aggregate completed scorecards |
