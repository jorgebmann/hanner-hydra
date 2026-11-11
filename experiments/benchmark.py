"""Resampling benchmark harness (pre-registered protocol): accuracy over 30 resamples.

Resamples follow the tsml-eval / bake-off convention used by the published Hydra results:

* resample 0 is the default train/test split
* resample ``i > 0`` pools train and test and redraws a stratified split with the original
  class counts (``aeon.benchmarking.resampling.stratified_resample_data``, seed ``i``)
* every arm is seeded with the resample id

Arms: ``hanner``, ``control_sym``, ``control`` (``make_classifier``, Numba backend when
available) and ``hydra`` (aeon ``HydraClassifier``, the M0 baseline). Each job is one
(dataset, resample) pair and runs all requested arms on the same split in one
single-threaded worker process. Jobs run in parallel with ``--jobs``.

One row per (dataset, resample, arm) is appended to ``results/benchmark/<tag>/results.csv``
(the results bundle), so an interrupted run resumes where it stopped. The test predictions go
to ``predictions/<dataset>/r<resample>/<arm>.csv`` (``y_true,y_pred``; the arm spec is
URL-quoted), written before the row. ``config.json`` in the same folder records the
settings; resuming with different settings is refused.

The recorded fit/predict times come from parallel workers and are **not** the timing
endpoint (that is ``experiments/timing.py`` and, later, the controlled timing harness).

Unequal-length datasets (4 of the 30 UEA datasets) are not supported yet and are skipped
with a message: see open question Q8 in ``docs/development-log.md``.

Usage:
    uv run python experiments/benchmark.py --suite ucr112 --resamples 30 --jobs 8
    uv run python experiments/benchmark.py ArrowHead Beef --resamples 5 --tag pilot
"""

from __future__ import annotations

import os

# Single-threaded workers: pin threads before numpy / torch / numba are imported. Spawned
# worker processes re-import this module, so the pin applies to them too.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import ast
import csv
import json
import platform
import time
import urllib.parse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_ROOT = ROOT / "results" / "benchmark"
TERNARY_ARMS = ("hanner", "control_sym", "control")
ARMS = (*TERNARY_ARMS, "hydra")
FIELDS = [
    "dataset",
    "resample",
    "arm",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "fit_s",
    "predict_s",
    "n_train",
    "n_test",
    "n_channels",
    "length",
]


def suite(name: str) -> list[str]:
    from aeon.datasets import tsc_datasets as t

    suites = {
        "ucr112": t.univariate_equal_length,  # the Hydra paper's 112 UCR datasets
        "uea30": t.multivariate,
        "uea26": t.multivariate_equal_length,
    }
    return sorted(suites[name])


def load(name: str):
    from aeon.datasets import load_classification

    X_train, y_train = load_classification(name, split="train", extract_path=DATA_DIR)
    X_test, y_test = load_classification(name, split="test", extract_path=DATA_DIR)
    if not isinstance(X_train, np.ndarray) or not isinstance(X_test, np.ndarray):
        raise ValueError(f"{name}: unequal-length series are not supported (open question Q8)")
    return X_train, y_train, X_test, y_test


def split(name: str, resample: int):
    from aeon.benchmarking.resampling import stratified_resample_data

    X_train, y_train, X_test, y_test = load(name)
    if resample > 0:
        X_train, y_train, X_test, y_test = stratified_resample_data(
            X_train, y_train, X_test, y_test, random_state=resample
        )
    return X_train, y_train, X_test, y_test


def parse_arm(spec: str) -> tuple[str, dict]:
    """``"hanner:max_vertices=18,center=False"`` -> ``("hanner", {...})``.

    The part before ``:`` is a base arm. The keyword arguments go to ``make_classifier``
    (ternary arms, overriding ``--features`` / ``--atom-budget``) or to aeon's
    ``HydraClassifier`` (e.g. ``hydra:n_kernels=24,n_groups=22``). Values are Python
    literals. The full spec string is the arm name in the results.
    """
    base, _, rest = spec.partition(":")
    if base not in ARMS:
        raise ValueError(f"unknown arm {base!r} in {spec!r} (choose from {ARMS})")
    kwargs = {}
    for item in filter(None, rest.split(",")):
        key, eq, value = item.partition("=")
        if not eq:
            raise ValueError(f"expected key=value, got {item!r} in {spec!r}")
        try:
            kwargs[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            kwargs[key.strip()] = value.strip()  # bare string, e.g. features=hydra
    return base, kwargs


def make(arm: str, seed: int, config: dict):
    base, kwargs = parse_arm(arm)
    if base == "hydra":
        import torch
        from aeon.classification.convolution_based import HydraClassifier

        torch.set_num_threads(1)
        return HydraClassifier(random_state=seed, n_jobs=1, **kwargs)
    from hanner_hydra import make_classifier

    params = {"features": config["features"], "atom_budget": config["atom_budget"]}
    return make_classifier(kernel_family=base, **{**params, **kwargs}, random_state=seed)


def run_job(name: str, resample: int, arms: list[str], config: dict) -> list[dict]:
    """All requested arms on one (dataset, resample) split; returns one row per arm.

    Each row carries the test labels and predictions under ``"_pred"`` (popped by the
    main process, which writes them to the predictions folder).
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    X_train, y_train, X_test, y_test = split(name, resample)
    rows = []
    for arm in arms:
        clf = make(arm, resample, config)
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        t1 = time.perf_counter()
        pred = clf.predict(X_test)
        t2 = time.perf_counter()
        rows.append(
            {
                "dataset": name,
                "resample": resample,
                "arm": arm,
                "accuracy": float(np.mean(pred == y_test)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "f1_macro": float(f1_score(y_test, pred, average="macro")),
                "fit_s": round(t1 - t0, 3),
                "predict_s": round(t2 - t1, 3),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_channels": X_train.shape[1] if X_train.ndim == 3 else 1,
                "length": X_train.shape[-1],
                "_pred": (np.asarray(y_test), np.asarray(pred)),
            }
        )
    return rows


def write_predictions(out_dir: Path, row: dict) -> None:
    y_true, y_pred = row.pop("_pred")
    folder = out_dir / "predictions" / row["dataset"] / f"r{row['resample']}"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{urllib.parse.quote(row['arm'], safe='')}.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_true", "y_pred"])
        w.writerows(zip(y_true.tolist(), y_pred.tolist(), strict=True))


def warm_up(config: dict) -> None:
    """Compile (and cache) the Numba kernels once before the workers start."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 2, 40))
    y = np.array([0, 1] * 3)
    for arm in TERNARY_ARMS:
        make(arm, 0, config).fit(X, y).predict(X)


def transform_defaults() -> dict:
    from hanner_hydra import HannerHydraTransform

    return {k: v for k, v in HannerHydraTransform().get_params().items() if k != "random_state"}


def check_config(out_dir: Path, config: dict) -> None:
    path = out_dir / "config.json"
    if path.exists():
        old = json.loads(path.read_text())
        keys = ("features", "atom_budget", "transform_defaults")
        if any(old.get(k) != config[k] for k in keys):
            raise SystemExit(
                f"{path} was written with different settings "
                f"({ {k: old.get(k) for k in keys} }); use another --tag"
            )
        return
    path.write_text(json.dumps(config, indent=2) + "\n")


def done_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise SystemExit(f"{path} has columns {reader.fieldnames}, expected {FIELDS}")
        return {(r["dataset"], int(r["resample"]), r["arm"]) for r in reader}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="*", help="dataset names (or use --suite)")
    parser.add_argument("--suite", choices=["ucr112", "uea30", "uea26"])
    parser.add_argument("--arms", nargs="+", default=list(ARMS), help="base arm[:key=value,...]")
    parser.add_argument("--resamples", type=int, default=30, help="resample ids 0..n-1")
    parser.add_argument("--first-resample", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--features", choices=["max", "hydra"], default="max")
    parser.add_argument("--atom-budget", type=int, default=256)
    parser.add_argument("--tag", default="default", help="output folder under results/benchmark")
    args = parser.parse_args()

    names = list(args.datasets) + (suite(args.suite) if args.suite else [])
    if not names:
        parser.error("give dataset names or --suite")
    resamples = range(args.first_resample, args.first_resample + args.resamples)

    out_dir = RESULTS_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    config = {
        "features": args.features,
        "atom_budget": args.atom_budget,
        # A tag never mixes results produced under different transform defaults.
        "transform_defaults": transform_defaults(),
        "hanner_hydra": version("hanner-hydra"),
        "aeon": version("aeon"),
        "numpy": version("numpy"),
        "machine": f"{platform.machine()} {platform.processor() or platform.platform()}",
    }
    check_config(out_dir, config)
    out_csv = out_dir / "results.csv"
    done = done_keys(out_csv)
    try:
        bases = {parse_arm(a)[0] for a in args.arms}
        for a in args.arms:
            make(a, 0, config)  # validates the keyword arguments before any work starts
    except (ValueError, TypeError) as e:
        parser.error(str(e))
    if bases & set(TERNARY_ARMS):
        warm_up(config)

    plan: dict[str, list[tuple[int, list[str]]]] = {}
    for name in names:
        todo = [(r, [a for a in args.arms if (name, r, a) not in done]) for r in resamples]
        if todo := [(r, arms) for r, arms in todo if arms]:
            plan[name] = todo
    n_jobs = sum(len(t) for t in plan.values())
    n_done = 0
    print(f"{n_jobs} jobs ({len(plan)} datasets, arms {args.arms}) with {args.jobs} workers")

    new_file = not out_csv.exists()
    t_start = time.perf_counter()
    with (
        out_csv.open("a", newline="") as f,
        # No max_tasks_per_child: on Python 3.12 worker recycling deadlocked the pool
        # (pilot run 3: all workers gone, main process waiting forever).
        ProcessPoolExecutor(max_workers=args.jobs) as pool,
    ):
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        pending: dict[Future, tuple[str, int]] = {}

        def drain(block_until: int) -> None:
            nonlocal n_done
            while len(pending) > block_until:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    name, r = pending.pop(fut)
                    n_done += 1
                    try:
                        rows = fut.result()
                    except Exception as e:  # report and continue with the other jobs
                        print(f"[{n_done}/{n_jobs}] {name} r{r}: FAILED {e}", flush=True)
                        continue
                    for row in rows:
                        write_predictions(out_dir, row)
                    writer.writerows(rows)
                    f.flush()
                    accs = "  ".join(f"{row['arm']}={row['accuracy']:.4f}" for row in rows)
                    mins = (time.perf_counter() - t_start) / 60
                    print(f"[{n_done}/{n_jobs}] {name} r{r}  {accs}  ({mins:.1f} min)", flush=True)

        for name, todo in plan.items():
            try:
                load(name)  # download in the main process, so workers never race on it
            except ValueError as e:
                print(f"skip {e}", flush=True)
                n_jobs -= len(todo)
                continue
            for r, arms in todo:
                pending[pool.submit(run_job, name, r, arms, config)] = (name, r)
            drain(block_until=4 * args.jobs)  # keep the queue short (memory)
        drain(block_until=0)
    print(f"done: {n_done} jobs in {(time.perf_counter() - t_start) / 60:.1f} min -> {out_csv}")


if __name__ == "__main__":
    main()
