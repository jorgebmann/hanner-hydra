"""Peak transform memory per (dataset, arm) (docs/preregistration.md §6, secondary).

Each measurement runs in a fresh process with one thread:

1. load ``X_train`` and ``X_test`` from ``.npy`` copies (few temporary allocations)
2. ``fit`` the transform (the arms of ``experiments/timing.py``, seed 0) and run one
   untimed warm-up ``transform`` on two test cases (JIT compilation, torch start-up)
3. read the peak resident set size (``ru_maxrss``), call ``transform(X_test)``, read it
   again

``extra_mb`` is the rise of the peak: the memory ``transform(X_test)`` needed beyond what
the process already held. It includes the output array (``output_mb``). It measures the
process (NumPy, Numba and torch allocations alike), so the allocator's reuse of freed
pages can make it an underestimate; the same procedure applies to every arm.

Rows go to ``results/memory/<tag>/memory.csv`` (resumable).

Usage:
    uv run python experiments/memory.py --suite ucr112 --suite uea26 --jobs 3 --tag endpoint
    uv run python experiments/memory.py --tag endpoint --summary
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import json
import resource
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timing import DEFAULT_ARMS, load, make, suite

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "memory"
CACHE = RESULTS_ROOT / "npy"
FIELDS = [
    "dataset",
    "arm",
    "base_mb",
    "peak_mb",
    "extra_mb",
    "output_mb",
    "n_test",
    "n_channels",
    "length",
]


def peak_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def measure(name: str, arm: str, seed: int) -> dict:
    X_train = np.load(CACHE / f"{name}_train.npy")
    X_test = np.load(CACHE / f"{name}_test.npy")
    tr = make(arm, seed)
    tr.fit(X_train)  # type: ignore[attr-defined]
    tr.transform(X_test[:2])  # type: ignore[attr-defined]  # warm-up
    base = peak_mb()
    out = np.asarray(tr.transform(X_test))  # type: ignore[attr-defined]
    peak = peak_mb()
    return {
        "dataset": name,
        "arm": arm,
        "base_mb": round(base, 1),
        "peak_mb": round(peak, 1),
        "extra_mb": round(peak - base, 1),
        "output_mb": round(out.nbytes / 2**20, 1),
        "n_test": X_test.shape[0],
        "n_channels": X_test.shape[1],
        "length": X_test.shape[2],
    }


def summary(path: Path) -> None:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    by = {(r["dataset"], r["arm"]): r for r in rows}
    arms = sorted({r["arm"] for r in rows}, key=DEFAULT_ARMS.index)
    names = sorted(n for n in {r["dataset"] for r in rows} if all((n, a) in by for a in arms))
    print(f"{len(names)} datasets complete for {arms}")
    for a in arms:
        extra = np.array([float(by[n, a]["extra_mb"]) for n in names])
        print(
            f"  {a:<12} extra peak: median {np.median(extra):8.1f} MB, max {extra.max():8.1f} MB "
            f"({names[int(extra.argmax())]})"
        )
    for a in arms:
        if a != "hanner":
            r = np.array(
                [
                    float(by[n, a]["extra_mb"]) / max(float(by[n, "hanner"]["extra_mb"]), 0.1)
                    for n in names
                ]
            )
            print(f"  {a} / hanner extra peak: median {np.median(r):.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--suite", action="append", choices=["ucr112", "uea26"], default=[])
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=2, help="measurements in parallel")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--measure", nargs=2, metavar=("DATASET", "ARM"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.measure:
        print(json.dumps(measure(*args.measure, args.seed)))
        return
    out_dir = RESULTS_ROOT / args.tag
    path = out_dir / "memory.csv"
    if args.summary:
        summary(path)
        return
    names = sorted(set(args.datasets).union(*(suite(s) for s in args.suite)))
    if not names:
        parser.error("give dataset names or --suite")
    out_dir.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in names:
        if not (CACHE / f"{name}_test.npy").exists():
            X_train, X_test = load(name)
            np.save(CACHE / f"{name}_train.npy", X_train)
            np.save(CACHE / f"{name}_test.npy", X_test)
    done = set()
    if path.exists():
        with path.open() as f:
            done = {(r["dataset"], r["arm"]) for r in csv.DictReader(f)}
    todo = [(n, a) for n in names for a in args.arms if (n, a) not in done]
    print(f"{len(todo)} measurements with {args.jobs} in parallel", flush=True)

    def run(n: str, a: str) -> dict:
        cmd = [sys.executable, __file__, "--measure", n, a, "--seed", str(args.seed)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(res.stdout.strip().splitlines()[-1])

    new_file = not path.exists()
    with path.open("a", newline="") as f, ThreadPoolExecutor(args.jobs) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        futures = {pool.submit(run, n, a): (n, a) for n, a in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            n, a = futures[fut]
            try:
                row = fut.result()
            except subprocess.CalledProcessError as e:
                print(f"[{i}/{len(todo)}] {n} {a}: FAILED {e.stderr[-300:]}", flush=True)
                continue
            writer.writerow(row)
            f.flush()
            print(f"[{i}/{len(todo)}] {n} {a} extra {row['extra_mb']} MB", flush=True)
    summary(path)


if __name__ == "__main__":
    main()
