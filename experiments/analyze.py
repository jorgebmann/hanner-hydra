"""Analyse a benchmark run: per-dataset means and the Bayesian signed-rank decision.

Reads ``results/benchmark/<tag>/results.csv`` (written by ``experiments/benchmark.py``).

* Only (dataset, resample) pairs that every analysed arm completed are used, and only
  datasets with at least ``--min-resamples`` such pairs (default: the most any dataset has).
* Unit of analysis: the mean accuracy over those resamples per dataset, in percentage points.
* The candidate (default ``hanner``) is compared with every other arm by the Bayesian
  signed-rank test with ROPE [-0.5, 0.5] (``hanner_hydra.stats``). H1 (accuracy part)
  holds against an arm if ``P(delta >= -0.5) >= 0.95``.
* ``--published`` additionally compares the ``hydra`` arm with aeon's published Hydra
  results on the same resample ids (the 30-resample version of the M0 reproduction gate).

``--subset`` restricts the analysis to the pre-registered secondary subsets
(docs/preregistration.md §6): ``ucr112``, ``uea26``, or ``fresh`` (the endpoint datasets
that did not inform any design choice, i.e. without ``DESIGN_DATASETS``).

Per-dataset means are written to ``dataset_means.csv`` next to the results
(``dataset_means_<subset>.csv`` with ``--subset``).

Usage:
    uv run python experiments/analyze.py --tag default [--published]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from hanner_hydra.stats import bayesian_signed_rank

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "benchmark"
PUBLISHED_CSV = RESULTS_ROOT / "published_hydra_ucr112_30.csv"
THRESHOLD = 0.95
# The 45 datasets that informed design choices in M0-M1 (preregistration §10).
DESIGN_DATASETS = frozenset(
    [
        "Adiac",
        "Beef",
        "Car",
        "CinCECGTorso",
        "Computers",
        "DistalPhalanxOutlineCorrect",
        "ECG5000",
        "EOGVerticalSignal",
        "Earthquakes",
        "FiftyWords",
        "Ham",
        "Haptics",
        "Herring",
        "InlineSkate",
        "InsectWingbeatSound",
        "Lightning2",
        "Lightning7",
        "Meat",
        "MedicalImages",
        "OliveOil",
        "Rock",
        "ScreenType",
        "SemgHandGenderCh2",
        "Strawberry",
        "SwedishLeaf",
        "Wine",
        "WordSynonyms",
        "Worms",
        "ArticularyWordRecognition",
        "Cricket",
        "ERing",
        "EthanolConcentration",
        "FingerMovements",
        "HandMovementDirection",
        "Handwriting",
        "Heartbeat",
        "LSST",
        "Libras",
        "NATOPS",
        "PEMS-SF",
        "PhonemeSpectra",
        "RacketSports",
        "SelfRegulationSCP1",
        "SelfRegulationSCP2",
        "UWaveGestureLibrary",
    ]
)


def subset(name: str) -> set[str]:
    from aeon.datasets import tsc_datasets as t

    ucr, uea = set(t.univariate_equal_length), set(t.multivariate_equal_length)
    return {"ucr112": ucr, "uea26": uea, "fresh": (ucr | uea) - DESIGN_DATASETS}[name]


def read(path: Path) -> dict[str, dict[tuple[str, int], float]]:
    """``{arm: {(dataset, resample): accuracy in points}}``."""
    acc: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            acc[r["arm"]][(r["dataset"], int(r["resample"]))] = 100 * float(r["accuracy"])
    return acc


def published() -> dict[tuple[str, int], float]:
    """aeon's published Hydra accuracies per (dataset, resample), downloaded once."""
    if not PUBLISHED_CSV.exists():
        from aeon.benchmarking.results_loaders import get_estimator_results

        res = get_estimator_results("Hydra", num_resamples=30)["Hydra"]
        with PUBLISHED_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "resample", "accuracy"])
            for name, v in sorted(res.items()):
                for i, a in enumerate(np.asarray(v, float)):
                    w.writerow([name, i, a])
    with PUBLISHED_CSV.open() as f:
        return {
            (r["dataset"], int(r["resample"])): 100 * float(r["accuracy"])
            for r in csv.DictReader(f)
        }


def dataset_means(
    acc: dict[str, dict[tuple[str, int], float]], arms: list[str], min_resamples: int | None
) -> tuple[list[str], dict[str, np.ndarray], dict[str, int]]:
    common = set.intersection(*(set(acc[a]) for a in arms))
    per_ds: dict[str, list[int]] = defaultdict(list)
    for name, r in sorted(common):  # sorted: set order varies per process (deviation D1)
        per_ds[name].append(r)
    need = min_resamples or max((len(v) for v in per_ds.values()), default=0)
    names = sorted(n for n, rs in per_ds.items() if len(rs) >= need)
    means = {
        a: np.array([np.mean([acc[a][(n, r)] for r in per_ds[n]]) for n in names]) for a in arms
    }
    return names, means, {n: len(per_ds[n]) for n in names}


def compare(label: str, a: np.ndarray, b: np.ndarray) -> None:
    d = a - b
    res = bayesian_signed_rank(a, b, rope=0.5)
    wins, ties, losses = int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())
    verdict = "NOT WORSE" if res.p_not_worse >= THRESHOLD else "not shown"
    print(
        f"  {label:<24} mean {a.mean():6.2f} vs {b.mean():6.2f}  diff {d.mean():+.2f}  "
        f"W/T/L {wins}/{ties}/{losses}  "
        f"P(left/rope/right) {res.p_left:.3f}/{res.p_rope:.3f}/{res.p_right:.3f}  "
        f"P(not worse) {res.p_not_worse:.3f}  {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tag", default="default")
    parser.add_argument("--candidate", default="hanner")
    parser.add_argument("--arms", nargs="+", help="arms to analyse (default: all in the file)")
    parser.add_argument("--min-resamples", type=int)
    parser.add_argument("--published", action="store_true", help="hydra arm vs aeon reference")
    parser.add_argument("--subset", choices=["ucr112", "uea26", "fresh"])
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / args.tag
    acc = read(out_dir / "results.csv")
    if args.subset:
        keep = subset(args.subset)
        acc = defaultdict(
            dict, {a: {k: v for k, v in d.items() if k[0] in keep} for a, d in acc.items()}
        )
    arms = args.arms or sorted(acc, key=lambda a: (a != args.candidate, a))
    missing = [a for a in arms if a not in acc]
    if missing:
        raise SystemExit(f"no results for arms {missing}")

    names, means, counts = dataset_means(acc, arms, args.min_resamples)
    if not names:
        raise SystemExit("no dataset has complete results for all arms")
    rs = sorted(set(counts.values()))
    print(
        f"{len(names)} datasets, {rs[0]}{'' if len(rs) == 1 else f'-{rs[-1]}'} resamples each, "
        f"arms {arms}"
    )
    suffix = f"_{args.subset}" if args.subset else ""
    with (out_dir / f"dataset_means{suffix}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_resamples", *arms])
        for i, n in enumerate(names):
            w.writerow([n, counts[n], *(f"{means[a][i]:.4f}" for a in arms)])

    if args.candidate in arms:
        print(f"{args.candidate} vs:")
        for other in arms:
            if other != args.candidate:
                compare(other, means[args.candidate], means[other])
        if len(names) < 10:
            print("  (fewer than 10 datasets: the posterior is dominated by the prior)")

    if args.published:
        if "hydra" not in acc:
            raise SystemExit("--published needs the hydra arm")
        pub = published()
        keys = [k for k in acc["hydra"] if k in pub]
        per_ds: dict[str, list[int]] = defaultdict(list)
        for n, r in keys:
            per_ds[n].append(r)
        ds = sorted(per_ds)
        ours = np.array([np.mean([acc["hydra"][(n, r)] for r in per_ds[n]]) for n in ds])
        ref = np.array([np.mean([pub[(n, r)] for r in per_ds[n]]) for n in ds])
        res = bayesian_signed_rank(ours, ref, rope=0.5)
        print(
            f"hydra (ours) vs published, {len(ds)} datasets, {len(keys)} resamples: "
            f"diff {np.mean(ours - ref):+.2f} pts, P(rope) {res.p_rope:.3f} "
            f"(equivalent if >= {THRESHOLD})"
        )


if __name__ == "__main__":
    main()
