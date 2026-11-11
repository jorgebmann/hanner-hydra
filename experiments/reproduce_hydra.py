"""M0 gate: reproduce published Hydra accuracies with aeon's HydraClassifier.

Reference: the Hydra results distributed with aeon's results loader (bake-off redux,
Middlehurst, Schäfer & Bagnall 2024; timeseriesclassification.com ReferenceResults),
112 UCR datasets x 30 resamples. Resample 0 is the default train/test split, so this
script runs HydraClassifier once per dataset on that split and compares it with the
published resample-0 accuracy.

Results are appended per dataset to ``results/reproduce_hydra/hydra_ucr112.csv``, so an
interrupted run resumes where it stopped. Pass ``--summary`` to only print the summary.

The summary also shows the per-dataset results published with the Hydra paper
(``angus924/hydra``, ``results/results_ucr112_hydra.csv``; mean over the same 30
resamples) for context. They average about 0.3 points above aeon's reference.

Gate (recorded in docs/development-log.md): mean accuracy difference (ours - published) within
+/-0.5 points over all datasets. Per-dataset differences are expected: a single
resample with a different random draw of kernels can move small datasets by several
points.

Usage:
    uv run --extra bench python experiments/reproduce_hydra.py [--threads 8] [DATASET ...]
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "reproduce_hydra"
OUT_CSV = OUT_DIR / "hydra_ucr112.csv"
REF_CSV = OUT_DIR / "reference_hydra_ucr112.csv"
# Per-dataset results published with the Hydra paper (mean over the same 30 resamples).
PAPER_URL = (
    "https://raw.githubusercontent.com/angus924/hydra/refs/heads/main/results/"
    "results_ucr112_hydra.csv"
)
PAPER_CSV = OUT_DIR / "paper_hydra_ucr112.csv"
FIELDS = ["dataset", "published_r0", "published_mean30", "ours", "diff", "fit_predict_s"]


def load_reference() -> dict[str, tuple[float, float]]:
    """``{dataset: (resample-0 accuracy, mean over 30 resamples)}``, cached as CSV."""
    if REF_CSV.exists():
        with REF_CSV.open() as f:
            return {r["dataset"]: (float(r["r0"]), float(r["mean30"])) for r in csv.DictReader(f)}
    from aeon.benchmarking.results_loaders import get_estimator_results

    res = get_estimator_results("Hydra", num_resamples=30)["Hydra"]
    ref = {k: (float(np.asarray(v)[0]), float(np.mean(v))) for k, v in sorted(res.items())}
    with REF_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "r0", "mean30"])
        for k, (r0, m) in ref.items():
            w.writerow([k, r0, m])
    return ref


def load_paper() -> dict[str, float]:
    """Hydra paper accuracies (mean over 30 resamples), downloaded once and cached."""
    if not PAPER_CSV.exists():
        import urllib.request

        with urllib.request.urlopen(PAPER_URL, timeout=30) as r:
            PAPER_CSV.write_bytes(r.read())
    with PAPER_CSV.open() as f:
        return {r["dataset"]: float(r["accuracy"]) for r in csv.DictReader(f)}


def done_rows() -> dict[str, dict[str, str]]:
    if not OUT_CSV.exists():
        return {}
    with OUT_CSV.open() as f:
        return {r["dataset"]: r for r in csv.DictReader(f)}


def run_dataset(name: str, seed: int) -> tuple[float, float]:
    from aeon.classification.convolution_based import HydraClassifier
    from aeon.datasets import load_classification

    X_train, y_train = load_classification(name, split="train", extract_path=DATA_DIR)
    X_test, y_test = load_classification(name, split="test", extract_path=DATA_DIR)
    t0 = time.perf_counter()
    clf = HydraClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    acc = float(np.mean(clf.predict(X_test) == y_test))
    return acc, time.perf_counter() - t0


def summarize(rows: dict[str, dict[str, str]]) -> None:
    if not rows:
        print("no results yet")
        return
    diff = np.array([float(r["diff"]) for r in rows.values()]) * 100
    ours = np.array([float(r["ours"]) for r in rows.values()]) * 100
    pub = np.array([float(r["published_r0"]) for r in rows.values()]) * 100
    print(f"datasets: {len(rows)}")
    print(f"mean accuracy  ours={ours.mean():.2f}  published(r0)={pub.mean():.2f}")
    print(
        f"mean diff {diff.mean():+.2f} pts  median {np.median(diff):+.2f}  "
        f"mean |diff| {np.abs(diff).mean():.2f}"
    )
    print(f"|diff| > 2 pts: {(np.abs(diff) > 2).sum()}   > 5 pts: {(np.abs(diff) > 5).sum()}")
    worst = sorted(rows.values(), key=lambda r: -abs(float(r["diff"])))[:8]
    for r in worst:
        print(
            f"  {r['dataset']:>28}  ours={float(r['ours']):.4f}  "
            f"published={float(r['published_r0']):.4f}  diff={float(r['diff']) * 100:+.2f}"
        )
    try:
        paper = load_paper()
        m30 = np.array([float(r["published_mean30"]) for r in rows.values()]) * 100
        pap = np.array([paper[k] for k in rows]) * 100
        print(
            f"context: aeon reference mean30={m30.mean():.2f}  Hydra paper mean30="
            f"{pap.mean():.2f}  (aeon ref - paper {np.mean(m30 - pap):+.2f} pts)"
        )
    except OSError as e:  # offline: the gate does not depend on the paper numbers
        print(f"(Hydra paper results unavailable: {e})")
    gate = abs(diff.mean()) <= 0.5
    print(f"GATE (|mean diff| <= 0.5 pts): {'PASS' if gate else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="subset (default: all 112)")
    parser.add_argument("--seed", type=int, default=0, help="tsml-eval convention: = resample")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = load_reference()
    if args.summary:
        summarize(done_rows())
        return

    import torch

    torch.set_num_threads(args.threads)  # accuracy run: timing is not an endpoint here
    names = args.datasets or list(ref)
    done = done_rows()
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, name in enumerate(names, 1):
            if name in done:
                continue
            r0, m30 = ref[name]
            acc, secs = run_dataset(name, args.seed)
            writer.writerow(
                {
                    "dataset": name,
                    "published_r0": r0,
                    "published_mean30": m30,
                    "ours": acc,
                    "diff": acc - r0,
                    "fit_predict_s": round(secs, 2),
                }
            )
            f.flush()
            print(
                f"[{i:>3}/{len(names)}] {name:>28}  ours={acc:.4f}  published={r0:.4f}  "
                f"diff={(acc - r0) * 100:+.2f}  ({secs:.1f}s)",
                flush=True,
            )
    summarize(done_rows())


if __name__ == "__main__":
    main()
