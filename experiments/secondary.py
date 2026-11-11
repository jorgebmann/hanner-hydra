"""Secondary analyses of the M2 run (docs/preregistration.md §6; reported, not deciding).

Reads ``results/benchmark/<tag>/results.csv`` and ``results/timing/<timing-tag>/`` and
writes tables and figures to ``results/benchmark/<tag>/secondary/``. Dataset means and the
Bayesian signed-rank test come from ``experiments/analyze.py`` unchanged.

* ``wilcoxon``: two-sided Wilcoxon signed-rank tests on dataset means for every pair of
  arms, Holm-corrected over the pairs, and a critical-difference diagram (Friedman test,
  Wilcoxon-Holm cliques, alpha 0.05; aeon ``plot_critical_difference``). Over all
  datasets and per subset.
* ``metrics``: balanced accuracy and macro-F1, analysed as accuracy (Bayesian signed-rank
  test, ROPE 0.5 points, Hanner against every arm).
* ``published``: Hanner-Hydra against aeon's published 30-resample results of WEASEL 2.0,
  MiniROCKET, MultiROCKET and HIVE-COTE 2.0 on the 112 UCR datasets, paired on dataset
  means (the resample splits differ), plus a CD diagram of all eight methods.
* ``fit``: fit times from the timing passes (median of the pass values per dataset).
* ``ablations``: the exploratory ablations of ``--tag ablations`` (``center`` False/True,
  ``features="hydra"``, ``max_channels`` 32/"auto") against default Hanner, ``control_sym``
  and Hydra.

Usage:
    uv run python experiments/secondary.py all --tag m2 --timing-tag endpoint
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import RESULTS_ROOT, THRESHOLD, dataset_means, subset
from hanner_hydra.stats import bayesian_signed_rank

ROOT = Path(__file__).resolve().parents[1]
TIMING_ROOT = ROOT / "results" / "timing"
ARMS = ["hanner", "control_sym", "control", "hydra"]
LABELS = {
    "hanner": "Hanner-Hydra",
    "control_sym": "control_sym",
    "control": "control",
    "hydra": "Hydra",
}
PUBLISHED = ["WEASEL-2.0", "MiniROCKET", "MultiROCKET", "HC2"]
ALPHA = 0.05


def read(path: Path, metric: str) -> dict[str, dict[tuple[str, int], float]]:
    """``{arm: {(dataset, resample): metric in points}}`` (``analyze.read`` for any metric)."""
    out: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f):
            out[r["arm"]][(r["dataset"], int(r["resample"]))] = 100 * float(r[metric])
    return out


def restrict(acc: dict, keep: set[str] | None) -> dict:
    if keep is None:
        return acc
    return defaultdict(
        dict, {a: {k: v for k, v in d.items() if k[0] in keep} for a, d in acc.items()}
    )


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def verdict(res) -> str:
    if res.p_right >= THRESHOLD:
        return "better"
    if res.p_rope >= THRESHOLD:
        return "practically equivalent"
    if res.p_not_worse >= THRESHOLD:
        return "not worse"
    if res.p_left >= THRESHOLD:
        return "practically worse"  # the mirror of "better" (P(delta < -0.5) >= 0.95)
    return "inconclusive"


def pairwise(names: list[str], means: dict[str, np.ndarray], arms: list[str]) -> list[dict]:
    rows = []
    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            d = means[a] - means[b]
            p = float(wilcoxon(means[a], means[b], alternative="two-sided").pvalue)
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "n": len(names),
                    "mean_diff": float(d.mean()),
                    "median_diff": float(np.median(d)),
                    "wins": int((d > 0).sum()),
                    "ties": int((d == 0).sum()),
                    "losses": int((d < 0).sum()),
                    "p": p,
                }
            )
    for r, p_adj in zip(rows, holm([r["p"] for r in rows]), strict=True):
        r["p_holm"] = p_adj
        r["significant"] = p_adj < ALPHA
    return rows


def cd_diagram(
    means: dict[str, np.ndarray], arms: list[str], labels: list[str], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from aeon.visualisation import plot_critical_difference

    scores = np.column_stack([means[a] for a in arms])
    fig, _ = plot_critical_difference(
        scores, labels, test="wilcoxon", correction="holm", alpha=ALPHA, width=7
    )
    fig.savefig(path, bbox_inches="tight", dpi=150)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def friedman(means: dict[str, np.ndarray], arms: list[str]) -> float:
    from scipy.stats import friedmanchisquare

    return float(friedmanchisquare(*(means[a] for a in arms)).pvalue)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def print_pairs(rows: list[dict]) -> None:
    for r in rows:
        star = "*" if r["significant"] else " "
        print(
            f"  {r['a']:>12} vs {r['b']:<12} diff {r['mean_diff']:+.2f} "
            f"(median {r['median_diff']:+.2f})  W/T/L {r['wins']}/{r['ties']}/{r['losses']}  "
            f"p {r['p']:.2g}  Holm {r['p_holm']:.2g} {star}"
        )


def run_wilcoxon(out: Path, results: Path) -> None:
    from scipy.stats import rankdata

    acc = read(results, "accuracy")
    for sub in (None, "ucr112", "uea26", "fresh"):
        names, means, _ = dataset_means(restrict(acc, subset(sub) if sub else None), ARMS, None)
        rows = pairwise(names, means, ARMS)
        tag = sub or "all"
        write_rows(out / f"wilcoxon_{tag}.csv", rows)
        cd_diagram(means, ARMS, [LABELS[a] for a in ARMS], out / f"cd_{tag}.png")
        ranks = rankdata(-np.column_stack([means[a] for a in ARMS]), axis=1).mean(axis=0)
        rank_txt = ", ".join(
            f"{a} {r:.2f}" for a, r in sorted(zip(ARMS, ranks, strict=True), key=lambda x: x[1])
        )
        print(
            f"[{tag}] {len(names)} datasets, Friedman p {friedman(means, ARMS):.2g}, "
            f"average ranks: {rank_txt}"
        )
        print_pairs(rows)


def run_metrics(out: Path, results: Path) -> None:
    rows = []
    for metric in ("accuracy", "balanced_accuracy", "f1_macro"):
        for sub in (None, "ucr112", "uea26", "fresh"):
            acc = restrict(read(results, metric), subset(sub) if sub else None)
            names, means, _ = dataset_means(acc, ARMS, None)
            for other in ARMS[1:]:
                res = bayesian_signed_rank(means["hanner"], means[other], rope=0.5)
                rows.append(
                    {
                        "metric": metric,
                        "subset": sub or "all",
                        "vs": other,
                        "n": len(names),
                        "hanner_mean": float(means["hanner"].mean()),
                        "other_mean": float(means[other].mean()),
                        "diff": float((means["hanner"] - means[other]).mean()),
                        "p_left": res.p_left,
                        "p_rope": res.p_rope,
                        "p_right": res.p_right,
                        "p_not_worse": res.p_not_worse,
                        "verdict": verdict(res),
                    }
                )
    write_rows(out / "metrics_bayes.csv", rows)
    for r in rows:
        if r["metric"] != "accuracy" and r["subset"] in ("all", "uea26"):
            print(
                f"  {r['metric']:>17} {r['subset']:>6} vs {r['vs']:<12} "
                f"{r['hanner_mean']:.2f} vs {r['other_mean']:.2f}  diff {r['diff']:+.2f}  "
                f"P(not worse) {r['p_not_worse']:.3f}  {r['verdict']}"
            )


def published_means(name: str, datasets: list[str]) -> dict[str, float]:
    path = RESULTS_ROOT / f"published_{name}_ucr112_30.csv"
    if not path.exists():
        from aeon.benchmarking.results_loaders import get_estimator_results

        res = get_estimator_results(name, datasets=datasets, num_resamples=30)[name]
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "resample", "accuracy"])
            for ds, v in sorted(res.items()):
                for i, a in enumerate(np.asarray(v, float)):
                    w.writerow([ds, i, a])
    per: dict[str, list[float]] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            per[r["dataset"]].append(100 * float(r["accuracy"]))
    return {ds: float(np.mean(v)) for ds, v in per.items() if len(v) == 30}


def run_published(out: Path, results: Path) -> None:
    names, means, _ = dataset_means(
        restrict(read(results, "accuracy"), subset("ucr112")), ARMS, None
    )
    pub = {m: published_means(m, names) for m in PUBLISHED}
    missing = {m: sorted(set(names) - set(v)) for m, v in pub.items() if set(names) - set(v)}
    if missing:
        raise SystemExit(f"published results missing for {missing}")
    allm = dict(means) | {m: np.array([pub[m][n] for n in names]) for m in PUBLISHED}
    rows = []
    for m in PUBLISHED:
        res = bayesian_signed_rank(allm["hanner"], allm[m], rope=0.5)
        d = allm["hanner"] - allm[m]
        rows.append(
            {
                "vs": m,
                "n": len(names),
                "hanner_mean": float(allm["hanner"].mean()),
                "other_mean": float(allm[m].mean()),
                "diff": float(d.mean()),
                "wins": int((d > 0).sum()),
                "ties": int((d == 0).sum()),
                "losses": int((d < 0).sum()),
                "p_left": res.p_left,
                "p_rope": res.p_rope,
                "p_right": res.p_right,
                "verdict": verdict(res),
            }
        )
    write_rows(out / "published_bayes.csv", rows)
    for r in rows:
        print(
            f"  Hanner-Hydra vs {r['vs']:<12} {r['hanner_mean']:.2f} vs {r['other_mean']:.2f}  "
            f"diff {r['diff']:+.2f}  W/T/L {r['wins']}/{r['ties']}/{r['losses']}  "
            f"P(left/rope/right) {r['p_left']:.3f}/{r['p_rope']:.3f}/{r['p_right']:.3f}  "
            f"{r['verdict']}"
        )
    arms = ARMS + PUBLISHED
    pairs = pairwise(names, allm, arms)
    write_rows(out / "wilcoxon_published_ucr112.csv", pairs)
    cd_diagram(allm, arms, [LABELS.get(a, a) for a in arms], out / "cd_published_ucr112.png")
    from scipy.stats import rankdata

    ranks = rankdata(-np.column_stack([allm[a] for a in arms]), axis=1).mean(axis=0)
    print(
        f"  CD (UCR, 8 methods), Friedman p {friedman(allm, arms):.2g}, average ranks: "
        + ", ".join(
            f"{a} {r:.2f}" for a, r in sorted(zip(arms, ranks, strict=True), key=lambda x: x[1])
        )
    )
    print_pairs([p for p in pairs if "hanner" in (p["a"], p["b"])])


def run_ablations(out: Path, results: Path, ablation_results: Path) -> None:
    """Exploratory ablations (``--tag ablations``), each against default Hanner and Hydra.

    ``max_channels`` only changes datasets with C >= 18 (elsewhere the transform is
    identical to the default, ``tests/test_transform.py``), so those arms ran on the 7
    such datasets and are compared there.
    """
    acc = read(results, "accuracy")
    abl = read(ablation_results, "accuracy")
    rows = []
    for arm in sorted(abl):
        both = defaultdict(dict, {**acc, arm: abl[arm]})
        names, means, _ = dataset_means(both, [arm, "hanner", "control_sym", "hydra"], None)
        for other in ("hanner", "control_sym", "hydra"):
            d = means[arm] - means[other]
            res = bayesian_signed_rank(means[arm], means[other], rope=0.5)
            rows.append(
                {
                    "arm": arm,
                    "vs": other,
                    "n": len(names),
                    "arm_mean": float(means[arm].mean()),
                    "other_mean": float(means[other].mean()),
                    "diff": float(d.mean()),
                    "wins": int((d > 0).sum()),
                    "ties": int((d == 0).sum()),
                    "losses": int((d < 0).sum()),
                    "p_left": res.p_left,
                    "p_rope": res.p_rope,
                    "p_right": res.p_right,
                    "verdict": verdict(res) if len(names) >= 10 else "n < 10, not tested",
                }
            )
    write_rows(out / "ablations.csv", rows)
    for r in rows:
        print(
            f"  {r['arm']:<26} vs {r['vs']:<12} n {r['n']:>3}  {r['arm_mean']:.2f} vs "
            f"{r['other_mean']:.2f}  diff {r['diff']:+.2f}  W/T/L {r['wins']}/{r['ties']}/"
            f"{r['losses']}  P(left/rope/right) {r['p_left']:.3f}/{r['p_rope']:.3f}/"
            f"{r['p_right']:.3f}  {r['verdict']}"
        )


def run_fit(out: Path, timing_dir: Path) -> None:
    config_passes = 3
    per: dict[tuple[str, str], list[float]] = defaultdict(list)
    for k in range(config_passes):
        with (timing_dir / f"pass{k}.csv").open() as f:
            for r in csv.DictReader(f):
                per[r["dataset"], r["arm"]].append(float(r["fit_s"]))
    names = sorted({n for n, _ in per})
    arms = ["hydra", "hanner", "control_sym", "control"]
    t = {(n, a): statistics.median(per[n, a]) for n in names for a in arms}
    rows = [{"dataset": n, **{f"{a}_fit_s": t[n, a] for a in arms}} for n in names]
    write_rows(out / "fit_times.csv", rows)
    for a in arms:
        tot = sum(t[n, a] for n in names)
        print(
            f"  {a:<12} fit: median {np.median([t[n, a] for n in names]):.3f} s, total {tot:.0f} s"
        )
    for a in ("hydra", "control_sym"):
        r = np.array([t[n, a] / t[n, "hanner"] for n in names])
        print(f"  {a} / hanner fit-time ratio: median {np.median(r):.2f}x (min {r.min():.2f}x)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "what", choices=["wilcoxon", "metrics", "published", "fit", "ablations", "all"]
    )
    parser.add_argument("--tag", default="m2")
    parser.add_argument("--timing-tag", default="endpoint")
    args = parser.parse_args()
    out = RESULTS_ROOT / args.tag / "secondary"
    out.mkdir(parents=True, exist_ok=True)
    results = RESULTS_ROOT / args.tag / "results.csv"
    steps = {
        "wilcoxon": lambda: run_wilcoxon(out, results),
        "metrics": lambda: run_metrics(out, results),
        "published": lambda: run_published(out, results),
        "fit": lambda: run_fit(out, TIMING_ROOT / args.timing_tag),
        "ablations": lambda: run_ablations(
            out, results, RESULTS_ROOT / "ablations" / "results.csv"
        ),
    }
    for name, step in steps.items():
        if args.what in (name, "all"):
            print(f"== {name}")
            step()
    print(f"-> {out}")


if __name__ == "__main__":
    main()
