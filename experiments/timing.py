"""Single-core transform timing: the timing endpoint (docs/preregistration.md §4).

Endpoint: per dataset the ratio Hydra time / Hanner-Hydra time of ``transform(X_test)``;
H1 needs the median ratio over the endpoint datasets to be >= 4.

Procedure (pre-registered):

* default train/test split (resample 0); every arm is ``fit`` on the training set
  (fit time recorded, not the endpoint) and seeded with ``--seed``
* one untimed warm-up ``transform(X_test)`` per arm (JIT compilation, caches)
* ``--repeats`` (7) timed ``transform(X_test)`` calls per arm, interleaved across arms in
  a rotating order, with ``gc.collect()`` before each call; per arm the median
* ``--passes`` (3) passes, each in its own process, each with a seeded random dataset order
* the time of an arm on a dataset is the median of its pass medians

Arms: ``hydra`` (aeon ``HydraTransformer(8, 64)``, public API: includes its float32 torch
conversion), ``hanner``, ``control_sym``, ``control`` (Numba kernel, float64). One thread
everywhere. Neither the scaler nor the ridge classifier is timed.

Per-pass rows go to ``results/timing/<tag>/pass<k>.csv`` (resumable per dataset);
``config.json`` records the settings and machine state, and resuming with different
settings is refused. ``--summary`` aggregates the completed datasets into
``summary.csv`` and prints the endpoint.

Usage:
    uv run python experiments/timing.py --suite ucr112 --suite uea26 --tag endpoint
    uv run python experiments/timing.py GunPoint NATOPS --passes 1 --repeats 3 --tag quick
    uv run python experiments/timing.py --tag endpoint --summary
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_ROOT = ROOT / "results" / "timing"
ARMS = ("hydra", "hanner", "control_sym", "control", "hanner_numpy")
DEFAULT_ARMS = ["hydra", "hanner", "control_sym", "control"]
FIELDS = [
    "pass",
    "dataset",
    "arm",
    "transform_median_s",
    "transform_times_s",
    "fit_s",
    "n_test",
    "n_channels",
    "length",
    "n_features",
]


def make(arm: str, seed: int) -> object:
    if arm == "hydra":
        import torch
        from aeon.transformations.collection.convolution_based import HydraTransformer

        torch.set_num_threads(1)
        return HydraTransformer(n_kernels=8, n_groups=64, random_state=seed, output_type="numpy")
    from hanner_hydra import HannerHydraTransform

    family, backend = ("hanner", "numpy") if arm == "hanner_numpy" else (arm, "numba")
    return HannerHydraTransform(kernel_family=family, backend=backend, random_state=seed)  # type: ignore[arg-type]


def suite(name: str) -> list[str]:
    from aeon.datasets import tsc_datasets as t

    return sorted({"ucr112": t.univariate_equal_length, "uea26": t.multivariate_equal_length}[name])


def load(name: str) -> tuple[np.ndarray, np.ndarray]:
    from aeon.datasets import load_classification

    X_train, _ = load_classification(name, split="train", extract_path=DATA_DIR)
    X_test, _ = load_classification(name, split="test", extract_path=DATA_DIR)
    return np.asarray(X_train, np.float64), np.asarray(X_test, np.float64)


def time_dataset(name: str, arms: list[str], repeats: int, seed: int) -> list[dict]:
    """Fit all arms, warm up, then ``repeats`` interleaved rounds of timed transforms."""
    X_train, X_test = load(name)
    fitted, fit_s, n_features = {}, {}, {}
    for arm in arms:
        tr = make(arm, seed)
        t0 = time.perf_counter()
        tr.fit(X_train)  # type: ignore[attr-defined]
        fit_s[arm] = time.perf_counter() - t0
        n_features[arm] = int(np.asarray(tr.transform(X_test)).shape[1])  # type: ignore[attr-defined]  # warm-up
        fitted[arm] = tr
    times: dict[str, list[float]] = {a: [] for a in arms}
    for rep in range(repeats):
        k = rep % len(arms)
        for arm in arms[k:] + arms[:k]:  # rotating order
            gc.collect()
            t0 = time.perf_counter()
            fitted[arm].transform(X_test)  # type: ignore[attr-defined]
            times[arm].append(time.perf_counter() - t0)
    return [
        {
            "dataset": name,
            "arm": arm,
            "transform_median_s": statistics.median(times[arm]),
            "transform_times_s": ";".join(f"{t:.6f}" for t in times[arm]),
            "fit_s": round(fit_s[arm], 6),
            "n_test": X_test.shape[0],
            "n_channels": X_test.shape[1],
            "length": X_test.shape[2],
            "n_features": n_features[arm],
        }
        for arm in arms
    ]


def machine_state() -> dict[str, str]:
    def sh(*cmd: str) -> str:
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"

    state = {"machine": f"{platform.machine()} {platform.processor() or platform.platform()}"}
    if sys.platform == "darwin":
        state["cpu"] = sh("sysctl", "-n", "machdep.cpu.brand_string")
        state["macos"] = platform.mac_ver()[0]
        state["power"] = sh("pmset", "-g", "ps").splitlines()[0] if sh("pmset", "-g", "ps") else ""
        low = [ln for ln in sh("pmset", "-g").splitlines() if "lowpowermode" in ln]
        state["lowpowermode"] = low[0].split()[-1] if low else "unknown"
    return state


def check_config(out_dir: Path, config: dict) -> None:
    path = out_dir / "config.json"
    keys = ("arms", "repeats", "passes", "seed", "order_seed", "datasets", "versions")
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != config[k] for k in keys):
            raise SystemExit(f"{path} was written with different settings; use another --tag")
        return
    path.write_text(json.dumps(config, indent=2) + "\n")


def run_pass(out_dir: Path, k: int, names: list[str], args: argparse.Namespace) -> None:
    path = out_dir / f"pass{k}.csv"
    done: set[str] = set()
    if path.exists():
        with path.open() as f:
            done = {r["dataset"] for r in csv.DictReader(f)}
    order = list(np.random.default_rng([args.order_seed, k]).permutation(names))
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, name in enumerate(order, 1):
            if name in done:
                continue
            rows = time_dataset(name, args.arms, args.repeats, args.seed)
            writer.writerows({"pass": k, **r} for r in rows)  # one dataset at a time: resumable
            f.flush()
            line = "  ".join(f"{r['arm']}={r['transform_median_s']:.3f}s" for r in rows)
            print(f"pass {k} [{i}/{len(order)}] {name}  {line}", flush=True)


def summary(out_dir: Path) -> None:
    config = json.loads((out_dir / "config.json").read_text())
    arms, passes = config["arms"], config["passes"]
    per: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for k in range(passes):
        path = out_dir / f"pass{k}.csv"
        if path.exists():
            with path.open() as f:
                for r in csv.DictReader(f):
                    per[r["dataset"], r["arm"]][k] = float(r["transform_median_s"])
    names = sorted(
        n for n in config["datasets"] if all(len(per.get((n, a), {})) == passes for a in arms)
    )
    if not names:
        raise SystemExit("no dataset is complete in all passes")
    t = {(n, a): statistics.median(per[n, a].values()) for n in names for a in arms}
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", *(f"{a}_s" for a in arms), "ratio_hydra_hanner"])
        for n in names:
            ratio = t[n, "hydra"] / t[n, "hanner"] if {"hydra", "hanner"} <= set(arms) else ""
            w.writerow([n, *(f"{t[n, a]:.6f}" for a in arms), ratio])
    print(f"{len(names)} of {len(config['datasets'])} datasets complete in all {passes} passes")
    if {"hydra", "hanner"} <= set(arms):
        ratios = np.array([t[n, "hydra"] / t[n, "hanner"] for n in names])
        rng = np.random.default_rng(0)
        boot = np.median(rng.choice(ratios, (10_000, ratios.size)), axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "MET" if np.median(ratios) >= 4 else "NOT MET"
        print(
            f"median Hydra / Hanner-Hydra ratio {np.median(ratios):.2f}x "
            f"(bootstrap 95% {lo:.2f}-{hi:.2f}, min {ratios.min():.2f}x)  criterion >= 4: {verdict}"
        )
        for k in range(passes):
            rk = [per[n, "hydra"][k] / per[n, "hanner"][k] for n in names]
            print(f"  pass {k}: median ratio {np.median(rk):.2f}x")
        for a in arms:
            if a not in ("hydra", "hanner"):
                r = np.median([t[n, a] / t[n, "hanner"] for n in names])
                print(f"  {a} / hanner: median {r:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--suite", action="append", choices=["ucr112", "uea26"], default=[])
    parser.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=ARMS)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0, help="seed of the transforms")
    parser.add_argument("--order-seed", type=int, default=0, help="seed of the dataset orders")
    parser.add_argument("--tag", default="default", help="output folder under results/timing")
    parser.add_argument("--summary", action="store_true", help="aggregate and print the endpoint")
    parser.add_argument("--pass-id", type=int, help=argparse.SUPPRESS)  # internal: one pass
    args = parser.parse_args()

    out_dir = RESULTS_ROOT / args.tag
    if args.summary:
        summary(out_dir)
        return
    names = sorted(set(args.datasets).union(*(suite(s) for s in args.suite)))
    if not names:
        parser.error("give dataset names or --suite")
    if args.pass_id is not None:
        run_pass(out_dir, args.pass_id, names, args)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    state = machine_state()
    config = {
        "arms": args.arms,
        "repeats": args.repeats,
        "passes": args.passes,
        "seed": args.seed,
        "order_seed": args.order_seed,
        "datasets": names,
        "versions": {p: version(p) for p in ("hanner-hydra", "aeon", "numpy", "numba", "torch")},
        **state,
    }
    check_config(out_dir, config)
    print(json.dumps(state))
    if "AC Power" not in state.get("power", "AC Power") or state.get("lowpowermode") == "1":
        print("WARNING: not on mains power or Low Power Mode is on (protocol violation)")
    for name in names:
        load(name)  # download before any pass, so no pass times a download
    for k in range(args.passes):
        cmd = [sys.executable, __file__, *names, "--pass-id", str(k), "--tag", args.tag]
        cmd += ["--arms", *args.arms, "--repeats", str(args.repeats)]
        cmd += ["--seed", str(args.seed), "--order-seed", str(args.order_seed)]
        subprocess.run(cmd, check=True)  # each pass in its own process
    summary(out_dir)


if __name__ == "__main__":
    main()
