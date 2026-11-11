"""Mechanism probe for long / smooth / drifting series: patch level energy and winner spread.

For each dataset (train split of resample 0, up to ``--cases`` cases):

* ``drift``: the share of patch energy that sits in the patch mean,
  ``sum(l * mean(p)^2) / sum(|p|^2)`` over all interior dilated patches of the raw
  channels, at dilation 1 and at the largest dilation. Close to 1 means the patches are
  dominated by their level (baseline offset or slow drift).
* per arm and input (series / difference), how spread out the hard counts are:
  ``top1`` is the mean share of anchors won by a group's most frequent atom, and
  ``eff`` is the effective number of winning atoms ``exp(entropy)`` as a fraction of
  the group size. ``top1`` near 1 means one atom wins almost everywhere, so the hard
  counts of that group carry little information.

Hydra's hard counts are its hard-min counts (8 kernels per group). The ternary arms use
hard-max counts (``features="max"``).

Usage:
    uv run python experiments/probe_drift.py Ham Meat Rock [--cases 40]
"""

from __future__ import annotations

import argparse

import numpy as np

from hanner_hydra import HannerHydraTransform

ARMS = {
    "hanner": {},
    "hanner:center=True": {"center": True},
    "hanner:increments=True": {"increments": True},
    "control_sym": {"kernel_family": "control_sym"},
}


def drift_index(X: np.ndarray, d: int, length: int = 9) -> float:
    span = (length - 1) * d + 1
    if X.shape[-1] < span:
        return float("nan")
    P = np.lib.stride_tricks.sliding_window_view(X, span, axis=-1)[..., ::d]
    m = P.mean(axis=-1)
    return float((length * m**2).sum() / (P**2).sum())


def spread(counts: np.ndarray) -> tuple[float, float]:
    """counts (n, G, k) -> (mean top-1 share, mean exp(entropy) / k)."""
    tot = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(tot == 0, 1, tot)
    top1 = p.max(axis=-1).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=-1)
    return float(top1), float((np.exp(ent) / counts.shape[-1]).mean())


def ternary_hard_counts(t: HannerHydraTransform, F: np.ndarray) -> dict[int, list[np.ndarray]]:
    """Split the feature matrix into per-group hard-count blocks, keyed by input."""
    out: dict[int, list[np.ndarray]] = {0: [], 1: []}
    c = 0
    for di in range(len(t.dilations_)):
        for inp in t.inputs_:
            for g in t.groups_[di, inp]:
                out[inp].append(F[:, c + g.size : c + 2 * g.size])
                c += 2 * g.size
    return out


def hydra_hard_counts(X: np.ndarray, seed: int) -> dict[int, list[np.ndarray]]:
    import torch
    from aeon.transformations.collection.convolution_based import HydraTransformer

    torch.set_num_threads(4)
    tr = HydraTransformer(random_state=seed, output_type="numpy").fit(X)
    F = np.asarray(tr.transform(X))
    h, k = tr._hydra.h, tr._hydra.k
    out: dict[int, list[np.ndarray]] = {0: [], 1: []}
    c = 0
    for _ in range(tr._hydra.num_dilations):
        for inp in range(tr._hydra.divisor):
            hard = F[:, c + h * k : c + 2 * h * k].reshape(len(X), h, k)
            out[inp].extend(hard[:, g] for g in range(h))
            c += 2 * h * k
    return out


def summarize(blocks: dict[int, list[np.ndarray]]) -> str:
    parts = []
    for inp, name in ((0, "raw"), (1, "diff")):
        if blocks[inp]:
            vals = [spread(b[:, None, :]) for b in blocks[inp]]
            top1, eff = np.mean(vals, axis=0)
            parts.append(f"{name}: top1 {top1:.2f} eff {eff:.2f}")
    return "  ".join(parts)


def main() -> None:
    from aeon.datasets import load_classification

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--cases", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for name in args.datasets:
        X, _ = load_classification(name, split="train", extract_path="data")
        X = np.asarray(X, np.float64)[: args.cases]
        dmax = 2 ** int(np.log2((X.shape[-1] - 1) / 8))
        print(
            f"{name}  C={X.shape[1]} T={X.shape[2]}  drift d=1 {drift_index(X, 1):.3f}  "
            f"d={dmax} {drift_index(X, dmax):.3f}"
        )
        print(f"  {'hydra':<24} {summarize(hydra_hard_counts(X, args.seed))}")
        for arm, kw in ARMS.items():
            t = HannerHydraTransform(random_state=args.seed, **kw).fit(X)
            print(f"  {arm:<24} {summarize(ternary_hard_counts(t, t.transform(X)))}")


if __name__ == "__main__":
    main()
