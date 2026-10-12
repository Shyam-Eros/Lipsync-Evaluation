#!/usr/bin/env python3
"""Aggregate completed human MOS scorecards and A/B preferences."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import config
from pipeline.io_utils import read_csv


def _f(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def bootstrap_ci(vals: List[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    if not vals:
        return (float("nan"), float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means)) - 1]
    return (sum(vals) / len(vals), lo, hi)


def aggregate_mos(scorecard_path: Path) -> Dict:
    rows = [r for r in read_csv(scorecard_path) if r.get("rater_id")]
    dims = ("sync_mos", "naturalness_mos", "quality_mos")
    by_model: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        model = r.get("model_name") or "unknown"
        for d in dims:
            v = _f(r.get(d))
            if v is not None:
                by_model[model][d].append(v)
    summary = {}
    for model, data in sorted(by_model.items()):
        summary[model] = {}
        for d in dims:
            mean, lo, hi = bootstrap_ci(data[d])
            summary[model][d] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n": len(data[d])}
    return {"mos_by_model": summary, "n_rated_rows": len(rows)}


def aggregate_ab(ab_path: Path) -> Dict:
    rows = [r for r in read_csv(ab_path) if r.get("rater_id") and r.get("winner")]
    wins_a = sum(1 for r in rows if str(r.get("winner")).upper() == "A")
    wins_b = sum(1 for r in rows if str(r.get("winner")).upper() == "B")
    n = wins_a + wins_b
    return {
        "n_pairs_rated": n,
        "wins_model_a": wins_a,
        "wins_model_b": wins_b,
        "win_rate_a": (wins_a / n) if n else None,
        "win_rate_b": (wins_b / n) if n else None,
    }


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--human-dir", type=Path, default=config.HUMAN_EVAL_DIR)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    human = Path(args.human_dir)
    mos_path = human / "mos_scorecard.csv"
    ab_path = human / "ab_pairs.csv"
    out = {
        "mos": aggregate_mos(mos_path) if mos_path.is_file() else {},
        "ab": aggregate_ab(ab_path) if ab_path.is_file() else {},
    }
    out_path = args.out or (human / "human_mos_summary.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    if out.get("mos", {}).get("mos_by_model"):
        for model, stats in out["mos"]["mos_by_model"].items():
            sm = stats.get("sync_mos", {})
            print(
                f"  {model}: sync_mos mean={sm.get('mean')} "
                f"[{sm.get('ci95_lo')}, {sm.get('ci95_hi')}] n={sm.get('n')}",
                flush=True,
            )
    if out.get("ab", {}).get("n_pairs_rated"):
        ab = out["ab"]
        print(
            f"  A/B: model_a wins {ab['wins_model_a']}, model_b wins {ab['wins_model_b']} "
            f"(n={ab['n_pairs_rated']})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
