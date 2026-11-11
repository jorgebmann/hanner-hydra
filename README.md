# Hanner-Hydra

**Hanner-polytope ternary kernel dictionaries for Hydra-style time series classification.**

> **Research question.** Can ternary Hanner-polytope kernel groups match `Hydra`'s accuracy
> at lower CPU cost?

## Idea in one paragraph

[Hydra](https://doi.org/10.1007/s10618-023-00939-3) organizes random Gaussian convolution
kernels into groups of competing kernels and, at every time step, counts which kernel in a
group wins. Hanner-Hydra replaces each group with the vertex set of a **Hanner polytope**,
the equality cases of the Mahler conjecture (`openai/math` Family 087, Lean-formalized). In
their canonical embedding every vertex lies in `{-1, 0, 1}^l`. The support function
`h(x) = max_v <x, v>` follows the polytope's cotree, so the **winning kernel of a whole group
costs O(l) additions and comparisons, with no multiplications**:

| Cotree node | Support function | Winner |
| :--- | :--- | :--- |
| leaf `[-1, 1]` on tap *i* | `abs(x_i)` | `sign(x_i)` |
| ℓ∞ product `K1 × K2` | `h1 + h2` | both winners, concatenated |
| ℓ1 sum `K1 ⊕₁ K2` | `max(h1, h2)` | winner of the larger child |

An unstructured group of the same size costs `O(|V| · l)` multiply-adds per patch. There
are two **theory-matched controls**, both random ternary groups with the same sizes and
per-vertex sparsity:

- `control` draws its vertices independently. This was the control of the original design.
- `control_sym` draws them in antipodal pairs `(u, -u)`, so the group is centrally
  symmetric like every Hanner polytope.

Together they separate the effect of the cotree structure from ternary weights and from
central symmetry. Because Hanner groups are symmetric, Hydra's "hard-min" counts would only
repeat the hard-max counts. The default features are therefore soft-max and hard-max counts
(`features="max"`); `features="hydra"` reproduces Hydra's choice for ablations.

## Hypothesis

Pre-registered in [docs/preregistration.md](docs/preregistration.md), which takes precedence
over this summary.

- **H1:** Hanner-Hydra is *not worse* than both Hydra and the symmetric control `control_sym`
  (ROPE ±0.5 accuracy points, Bayesian signed-rank test), **and** its single-core CPU
  transform is at least 4× faster than Hydra's (median ratio across datasets).
- **Benchmarks:** the 112 UCR datasets of the Hydra paper and the 26 equal-length UEA
  multivariate datasets, 30 stratified resamples each.
- **Secondary endpoints:** peak memory, balanced accuracy and macro-F1, and accuracy against
  WEASEL 2.0, MiniROCKET, MultiROCKET and HIVE-COTE 2.0.

## Status

**H1 is accepted** ([details](docs/development-log.md#h1-decision-2026-11-07)).
Over the 138 pre-registered datasets (112 UCR and 26 UEA, 30 resamples each), Hanner-Hydra
is not worse than Hydra in accuracy (Δ −0.06 points, P(not worse) 0.988) and not worse than
the theory-matched control `control_sym` (Δ +0.04, P 1.000). Its single-core transform is
**11.2× faster** than Hydra's (median ratio, Apple M4) and needs 24× less memory.

Caveats:
- It is equivalent to Hydra, not better.
- On the UEA datasets alone, "not worse than Hydra" is not shown.
- The cotree structure contributes speed (3.8× over the matched controls), not accuracy.

The paper is in preparation ([`paper/`](paper/)). The full record of the design, the pilots
and every decision is in [docs/development-log.md](docs/development-log.md).

- [x] aeon-compatible estimators `HannerHydraTransformer` and `HannerHydraClassifier`
  (`hanner_hydra.estimators`). They pass aeon's estimator checks and reproduce the
  reference transform bit for bit.
- [x] Results and scripts: the GitHub release [`results-v1`](https://github.com/jorgebmann/hanner-hydra/releases/tag/results-v1) holds
  `hanner-hydra-results-v1.tar.gz` (built by `experiments/release.py`): per-resample
  results, per-case predictions, timing, memory, secondary analyses and ablations, under
  CC BY 4.0, archived on Zenodo as
  [10.5281/zenodo.24142209](https://doi.org/10.5281/zenodo.24142209). Check out the tag, extract the archive in the repository root and the analysis
  scripts run on it unchanged ([contents](docs/results-release.md)).

Earlier steps:

- [x] Cotree sampler with a vertex budget `V_max` (no rejection sampling), plus a variant with contiguous tap blocks
- [x] Support-function recursion that returns the maximum and the winning vertex index; tested against brute force, including ties
- [x] Matched random-ternary controls: independent and centrally symmetric
- [x] Hydra-style transform as a NumPy reference: dilations, difference input, multivariate channel subsets, soft-max/hard-min counts, `SparseScaler`, ridge classifier
- [x] Pilot script against aeon's `HydraTransformer`
- [x] Numba backend (`backend="numba"`, the default when numba is installed), bit-identical to the reference
- [x] Single-core timing script against aeon's Hydra (`experiments/timing.py`)
- [x] Blocked node-major kernel (v2): median **8.9×** faster than Hydra on six datasets, at least 5× on each ([details](docs/development-log.md#early-timing-results-2026-11-03)). The timing endpoint is measured on this kernel, not on the NumPy reference.
- [x] Controlled timing harness for the endpoint (`experiments/timing.py`: Apple M4, one thread, interleaved repeats, three passes)
- [x] Primary endpoint pre-registered and frozen ([docs/preregistration.md](docs/preregistration.md), 2026-11-04)
- [x] Hydra baseline reproduced on the 112 UCR datasets (`experiments/reproduce_hydra.py`): mean difference +0.42 points versus published, gate ±0.5 passed ([details](docs/development-log.md#hydra-baseline-reproduction-2026-11-03))
- [x] Benchmark harness: 30 stratified resamples (`experiments/benchmark.py`) and the Bayesian signed-rank analysis (`experiments/analyze.py`, `hanner_hydra.stats`), cross-checked against `baycomp` ([details](docs/development-log.md#resampling-harness-2026-11-03))

## Quickstart

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync                                   # core: numpy + scikit-learn
uv run pytest                             # tests

uv sync --extra aeon --extra fast         # + aeon estimators, numba kernels
uv sync --extra bench --extra fast        # + aeon, torch (Hydra baseline), numba
uv run python experiments/pilot.py BasicMotions Epilepsy GunPoint
uv run python experiments/timing.py GunPoint NATOPS --passes 1 --tag try   # single-core timing
uv run python experiments/benchmark.py ArrowHead Beef --resamples 5 --tag try
uv run python experiments/analyze.py --tag try
```

The pilot downloads datasets into `data/` and writes a CSV to `results/pilot/`. Neither
folder is committed.

```python
import numpy as np
from hanner_hydra import make_classifier, sample_cotree

tree = sample_cotree(9, np.random.default_rng(0), max_vertices=32)
print(tree)  # Cotree(length=9, size=..., ((x4 x x1) +1 ...))
h, winner = tree.support(np.random.randn(5, 9))

clf = make_classifier(kernel_family="hanner", random_state=0)  # or "control", "control_sym"
clf.fit(X_train, y_train)  # X: (n_cases, n_timepoints) or (n_cases, C, T)
clf.score(X_test, y_test)
```

With aeon installed (`--extra aeon`), the same classifier and transform are available as
aeon estimators, with the frozen defaults:

```python
from aeon.datasets import load_classification
from hanner_hydra import HannerHydraClassifier, HannerHydraTransformer

X_train, y_train = load_classification("GunPoint", split="train")
X_test, y_test = load_classification("GunPoint", split="test")
clf = HannerHydraClassifier(random_state=0).fit(X_train, y_train)
print((clf.predict(X_test) == y_test).mean())

features = HannerHydraTransformer(random_state=0).fit_transform(X_train)  # (50, n_features)
```

## Repository layout

```
src/hanner_hydra/
  cotree.py       labeled cotrees: sampling, vertices, antipodes, support recursion
  control.py      theory-matched controls (random ternary groups; independent / symmetric)
  transform.py    HannerHydraTransform (NumPy reference + backend dispatch), SparseScaler
  estimators.py   aeon estimators: HannerHydraTransformer, HannerHydraClassifier
  _numba.py       single-threaded Numba kernels, bit-identical to the reference
  stats.py        Bayesian signed-rank test (Benavoli et al. 2017) for the endpoint
tests/            correctness tests (recursion vs. brute force, transform invariants)
experiments/      benchmark.py + analyze.py (30 resamples, Bayesian test), pilot.py, timing.py,
                  reproduce_hydra.py (Hydra baseline gate), secondary.py + memory.py (prereg §6),
                  release.py (results bundle)
docs/
  preregistration.md  the frozen protocol of the primary endpoint
  development-log.md  design history, pilots, results and decision log
  results-release.md  contents of the results bundle (its README)
paper/            paper draft (LaTeX outline, references.bib, literature-review.md)
data/             dataset cache (git-ignored)
results/          experiment outputs (git-ignored except .gitkeep)
```

## Source provenance

All `openai/math` references are pinned to commit `4eaee87` (October 30, 2026). Family 087
(*The symmetric Mahler conjecture and its equality cases*; *The Mahler Conjecture for General
Convex Bodies*, both dated September 22, 2026) is Lean-formalized, including the equality
cases. The theorem motivates the kernel family. It does not guarantee
any empirical performance; that is what the matched control is for.

## Key references

- Dempster, Schmidt & Webb (2023). Hydra: Competing convolutional kernels for fast and accurate time series classification. *DMKD* 37(5).
- Hanner (1956). Intersections of translates of convex bodies. *Math. Scand.* 4.
- Dempster, Schmidt & Webb (2021). MiniRocket. *KDD*.
- Middlehurst et al. (2021). HIVE-COTE 2.0. *Machine Learning* 110.

See [`paper/references.bib`](paper/references.bib) for the full list.

## License

The code is under [BSD-3-Clause](LICENSE), the same license as
[aeon](https://github.com/aeon-toolkit/aeon). The results bundle
(`experiments/release.py`) is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
see [its README](docs/results-release.md#licence) for how to cite it.
