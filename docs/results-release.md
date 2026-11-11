# Hanner-Hydra benchmark results

These are the per-resample results and per-case predictions of the Hanner-Hydra study.
They cover the pre-registered M2 endpoint (the full evaluation), its secondary analyses,
the ablations and the Hydra reproduction gate.

- Code: the `hanner-hydra` repository at the commit in `RELEASE.json`. The analysis
  scripts are in `experiments/`.
- Protocol: `docs/preregistration.md`, frozen on 2026-11-04 before any M2 run (its
  SHA-256 is in `RELEASE.json`). There is one later deviation, D1 in prereg §11. It fixes the resample order in the dataset means, which changed the W/T/L
  tie counts by one or two datasets and no decision.
- Write-up: `docs/development-log.md`, sections "M2 accuracy runs", "M2 timing endpoint",
  "H1 decision" and "M2 secondary analyses".

## Result: H1 accepted

| Criterion (prereg §5) | Result | Required |
| :--- | :--- | :--- |
| Accuracy: Hanner-Hydra not worse than Hydra | P(Δ ≥ −0.5) = 0.988 | ≥ 0.95 |
| Accuracy: Hanner-Hydra not worse than control_sym | P(Δ ≥ −0.5) = 1.000 | ≥ 0.95 |
| Median transform time ratio Hydra / Hanner-Hydra | 11.23× (bootstrap 95% 10.23–12.01) | ≥ 4 |

- Datasets: 112 UCR and 26 equal-length UEA datasets. Each has 30 stratified resamples
  (resample 0 is the default split).
- Statistical test: Bayesian signed-rank on the per-dataset mean accuracies in percentage
  points, ROPE ±0.5.
- Timing hardware: Apple M4, one thread.

## Using the bundle

Extract the bundle in the repository root of a fresh checkout of the release commit. This
recreates `results/`, and the scripts read it unchanged. Extracting into a checkout that
already has a `results/` folder overwrites the files of the same name; run
`tar tzf hanner-hydra-results-v1.tar.gz` first to see what it writes.

```sh
git checkout <commit from RELEASE.json>
tar xzf hanner-hydra-results-v1.tar.gz
uv sync --extra bench --extra fast
uv run python experiments/analyze.py --tag m2                  # H1 accuracy decision
uv run python experiments/timing.py --tag endpoint --summary   # H1 time ratio
uv run python experiments/memory.py --tag endpoint --summary
uv run python experiments/secondary.py all --tag m2 --timing-tag endpoint
```

None of these commands needs the UCR/UEA data. `analyze.py --published` and
`secondary.py published` download aeon's published results (see "Not included" below).

To check the bundle's integrity, compare the archive's SHA-256 with the one in the GitHub
release notes, then run `sha256sum -c MANIFEST.sha256` (`shasum -a 256 -c` on macOS)
inside `results/`, or `uv run python experiments/release.py --verify <archive>`. The
manifest covers every result file; `README.md` and `RELEASE.json` are covered only by the
archive checksum.

The CSV and PNG outputs of the scripts above are byte-identical to the bundled ones. The
regenerated CD-diagram PDFs differ only in their embedded creation date.

## Contents

All accuracies in the result files are fractions in [0, 1]. The analyses report
percentage points. All times are wall-clock seconds.

Arms:
- `hydra`: aeon 1.6.0 `HydraClassifier`, the baseline.
- `hanner`: Hanner-Hydra with the frozen defaults.
- `control`: the theory-matched control without the Hanner structure.
- `control_sym`: the same control, drawn in antipodal pairs.

Ablation arms are named `hanner:<parameter>=<value>`.

### `benchmark/m2/`: M2 accuracy runs

| File | Content |
| :--- | :--- |
| `config.json` | Transform defaults and package versions |
| `results.csv` | One row per (dataset, resample, arm), 16,560 rows. Columns: `accuracy`, `balanced_accuracy`, `f1_macro`, `fit_s` (fit time), `predict_s` (predict time), `n_train`, `n_test`, `n_channels`, `length` |
| `predictions/<dataset>/r<k>/<arm>.csv` | Test-set predictions per case. Columns: `y_true`, `y_pred` |
| `dataset_means*.csv` | Mean accuracy per dataset (percentage points) over the resamples every arm completed. Variants: all 138 datasets; `_ucr112`; `_uea26`; `_fresh` (the 93 datasets that did not inform any design choice) |
| `secondary/wilcoxon_<subset>.csv` | Pairwise two-sided Wilcoxon signed-rank tests with Holm step-down correction, α = 0.05 |
| `secondary/cd_<subset>.{png,pdf}` | Critical-difference diagrams. They use aeon's convention, which is not the pre-registered test (see `docs/development-log.md`) |
| `secondary/metrics_bayes.csv` | Bayesian signed-rank tests on accuracy, balanced accuracy and macro-F1, for each subset and each comparator |
| `secondary/published_bayes.csv`, `wilcoxon_published_ucr112.csv`, `cd_published_ucr112.*` | Comparison with aeon's published WEASEL 2.0, MiniROCKET, MultiROCKET and HC2 results (UCR, per-dataset means over 30 resamples) |
| `secondary/fit_times.csv` | Median fit time per dataset and arm, over the 3 timing passes (`timing/endpoint/pass<k>.csv`) |
| `secondary/ablations.csv` | Bayesian signed-rank tests of each ablation arm against `hanner` |

### `benchmark/ablations/`

This directory has the same `results.csv` and `predictions/` layout as M2.

- `center=False`, `center=True` and `features=hydra` ran on all 138 datasets × 30
  resamples.
- `max_channels=32` and `max_channels=auto` ran on the 7 datasets with more than 16
  channels.

### `timing/endpoint/`: M2 timing endpoint (prereg §4)

| File | Content |
| :--- | :--- |
| `config.json` | Arms, repeats, passes, seeds, dataset list, versions, CPU, macOS version, power state |
| `pass<k>.csv` | One row per (dataset, arm) for pass k. `transform_times_s` holds the 7 interleaved repeats of `transform(X_test)`, separated by `;`. Also `transform_median_s` (their median), `fit_s` (fit time), `n_features` |
| `summary.csv` | Per dataset and arm, the median over the 3 passes of the per-pass medians, plus `ratio_hydra_hanner`. This is the H1 time endpoint |
| `pass0_nonidle_20261106.csv`, `summary_with_nonidle_pass0.csv` | The first pass 0, which ran while the Mac was in use and so broke the idle condition. Kept for transparency; the endpoint does not use it |

### `memory/endpoint/memory.csv`: peak transform memory (prereg §6)

Each measurement runs in a fresh process. Columns:
- `base_mb`: the peak resident set size after fit and a warm-up.
- `peak_mb`: the peak after `transform(X_test)`.
- `extra_mb`: `peak_mb − base_mb`. This includes the output array, whose size is `output_mb`.

See `experiments/memory.py` for the method and its limits.

### `reproduce_hydra/hydra_ucr112.csv`: Hydra reproduction gate

Our aeon `HydraClassifier` runs on the default UCR splits (resample 0), against aeon's
published accuracies.

Columns:
- `published_r0`: aeon's published resample-0 accuracy.
- `published_mean30`: aeon's published mean over 30 resamples.
- `ours`: our accuracy.
- `diff`: `ours − published_r0`.
- `fit_predict_s`: fit plus predict time.

## Not included

Third-party results are not redistributed. The scripts download them when they are
missing:

- `benchmark/published_*_ucr112_30.csv`: aeon's published results (tsml reference results)
  for Hydra, WEASEL 2.0, MiniROCKET, MultiROCKET and HC2.
- `reproduce_hydra/reference_hydra_ucr112.csv` and `paper_hydra_ucr112.csv`: aeon's and
  the Hydra paper's published Hydra accuracies.

`RELEASE.json` records the SHA-256 of the copies the analyses used. After the scripts have
downloaded them, `uv run python experiments/release.py --check-third-party` compares
them; a different checksum means the published results have changed upstream.

Run logs, pilots and diagnostics from before the pre-registration are not included. Neither are the UCR/UEA
datasets, which are available from timeseriesclassification.com. The prediction files do
contain their test labels (`y_true`); if you use them, please cite the UCR archive
(Dau et al. 2019) and the UEA archive (Bagnall et al. 2018).

## Licence

The results in this bundle are © 2026 Jörg Bahlmann and licensed under the Creative Commons
Attribution 4.0 International licence (CC BY 4.0,
<https://creativecommons.org/licenses/by/4.0/>). You may share and adapt them for any
purpose, provided you give credit, link to the licence and say if you made changes.
`RELEASE.json` records the licence as `"license": "CC-BY-4.0"`.

Please cite them as:

> Bahlmann, J. (2026). *Hanner-Hydra benchmark results* (version results-v1) [Data set].
> Zenodo. https://doi.org/10.5281/zenodo.24142209

The same archive is attached to the GitHub release
[`results-v1`](https://github.com/jorgebmann/hanner-hydra/releases/tag/results-v1).

The code that produced them, in the `hanner-hydra` repository, is under the BSD 3-Clause
licence (`LICENSE` there). The UCR/UEA test labels in the prediction files come from those
archives (see "Not included").
