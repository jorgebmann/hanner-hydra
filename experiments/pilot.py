"""Pilot comparison: Hanner-Hydra vs. matched ternary controls vs. Hydra (aeon).

Runs the default train/test split of each dataset (no resampling) and records accuracy
and single-threaded transform times. This is a smoke-level pilot for milestone M1, not
the pre-registered protocol (30 resamples, Bayesian signed-rank).

Timing caveat: the Hanner/control arms use the NumPy *reference* implementation, so
their times are NOT the timing endpoint (that needs the multiplier-free kernel).

Usage:
    uv run --extra bench python experiments/pilot.py BasicMotions Epilepsy GunPoint
"""

from __future__ import annotations

import os

# Single-core timing: pin threads before numpy / torch are imported.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from hanner_hydra import HannerHydraTransform
from hanner_hydra.transform import SparseScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "pilot"


def hydra_transform(seed: int) -> object:
    import torch
    from aeon.transformations.collection.convolution_based import HydraTransformer

    torch.set_num_threads(1)
    return HydraTransformer(n_kernels=8, n_groups=64, random_state=seed, output_type="numpy")


def make_arms(seed: int, atom_budget: int, features: str) -> dict[str, Callable[[], object]]:
    def ternary(family: str) -> Callable[[], object]:
        return lambda: HannerHydraTransform(
            kernel_family=family,  # type: ignore[arg-type]
            features=features,  # type: ignore[arg-type]
            atom_budget=atom_budget,
            random_state=seed,
        )

    return {
        "hanner": ternary("hanner"),
        "control": ternary("control"),
        "control_sym": ternary("control_sym"),
        "hydra": lambda: hydra_transform(seed),
    }


def run_arm(make: Callable[[], object], X_train, y_train, X_test, y_test) -> dict[str, float]:
    transform = make()
    t0 = time.perf_counter()
    F_train = np.asarray(transform.fit_transform(X_train))  # type: ignore[attr-defined]
    t_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    F_test = np.asarray(transform.transform(X_test))  # type: ignore[attr-defined]
    t_test = time.perf_counter() - t0

    scaler = SparseScaler().fit(F_train)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(scaler.transform(F_train), y_train)
    acc = clf.score(scaler.transform(F_test), y_test)
    return {
        "accuracy": float(acc),
        "n_features": int(F_train.shape[1]),
        "transform_train_s": t_train,
        "transform_test_s": t_test,
    }


def main() -> None:
    from aeon.datasets import load_classification

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+", help="UCR/UEA dataset names")
    parser.add_argument("--arms", nargs="+", default=["hanner", "control", "control_sym", "hydra"])
    parser.add_argument(
        "--features", choices=["max", "hydra"], default="max", help="ternary arms only"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--atom-budget", type=int, default=256, help="Hydra: 8 kernels x 32 groups per input"
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"pilot_{datetime.now():%Y%m%d_%H%M%S}.csv"
    arms = make_arms(args.seed, args.atom_budget, args.features)

    rows: list[dict[str, object]] = []
    for name in args.datasets:
        X_train, y_train = load_classification(name, split="train", extract_path=DATA_DIR)
        X_test, y_test = load_classification(name, split="test", extract_path=DATA_DIR)
        for arm in args.arms:
            res = run_arm(arms[arm], X_train, y_train, X_test, y_test)
            row = {"dataset": name, "arm": arm, "features": args.features, "seed": args.seed, **res}
            rows.append(row)
            print(
                f"{name:>24} {arm:>8}  acc={res['accuracy']:.4f}  "
                f"feat={res['n_features']:>6}  "
                f"t_train={res['transform_train_s']:.2f}s  t_test={res['transform_test_s']:.2f}s"
            )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
