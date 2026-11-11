# Hanner-Hydra: development log

The design history of Hanner-Hydra: milestones, diagnostics, pilots, the full evaluation
and the decision log, with the dates of the work. The work ran in phases: M0
infrastructure, M1 pilots and pre-registration, M2 the full evaluation, M3 the release.
The protocol of the primary endpoint is [preregistration.md](preregistration.md); where
the two differ, the pre-registration takes precedence.

## Primary endpoint and decision rule

**Frozen 2026-11-04 in [preregistration.md](preregistration.md) , which
takes precedence over this summary.**

- **Endpoint:** Top-1 accuracy on the 112 Hydra UCR datasets and the 26 equal-length UEA
  datasets (138, one pooled test), together with single-core CPU transform time on the
  Apple M4.
- **H1:** declared *not worse* than both Hydra and the symmetric control (`control_sym`),
  meaning `P(Δ ≥ -0.5) ≥ 0.95` under the Bayesian signed-rank test with ROPE [-0.5, 0.5].
  In addition, the median transform-time ratio Hydra / Hanner-Hydra must be at least 4.
- **Controls:** `control_sym` is the primary theory-matched control. The independent
  control (`control`, the control of the original design) is reported alongside it as a
  secondary comparison. This deviates from the original design; the deviation is stated in the
  pre-registration (§8).
- **Unit of analysis:** the mean over 30 stratified resamples per dataset (aeon protocol).
- The endpoint was pre-registered at M1, before the full evaluation.

## Milestones

### M0: Infrastructure

- [x] Package skeleton, tests and CI
- [x] Cotree sampler, support recursion, vertex enumeration, antipodes
- [x] Matched random-ternary control, plus the centrally symmetric variant `control_sym`
- [x] Feature set fixed: soft-max + hard-max counts (open question 1)
- [x] NumPy reference transform plus a pilot script against aeon's `HydraTransformer`
- [x] Reproduce Hydra's published accuracies with aeon's `HydraClassifier` (gate for M0):
      **pass**, see "Hydra baseline reproduction"
- [x] Fix the exact UCR subset of Dempster et al. (2023): the
      112 UCR datasets of the bake-off studies, same 30 resamples (paper §4.1)
- [x] Timing harness: `experiments/timing.py`, the pre-registered procedure
      (transform only, one thread, warm-up, 7 interleaved repeats, 3 passes in separate
      processes, Apple M4; preregistration §4)
- [ ] Peak memory (`tracemalloc` / RSS) and cache-miss measurement. Cache counters need
      `perf` on Linux x86; the Apple M4 dev machine has no easy equivalent.
- [x] Resampling harness: 30 stratified resamples, with per-resample results stored as CSV
      (feeds the results bundle): `experiments/benchmark.py` and `experiments/analyze.py`,
      see "Resampling harness"

### M1: Pilots and pre-registration

- [x] Fast kernel, first version in Numba (`backend="numba"`, `src/hanner_hydra/_numba.py`).
      It evaluates the recursion per anchor and interprets the flattened cotree arrays.
- [x] Equivalence test: the Numba output is bit-identical to the reference for all
      families, both feature sets, with and without centering, and on tie-heavy input
- [x] Optimize the Hanner kernel for long series: blocked node-major v2, median 8.9×
      on six datasets (see "Early timing results")
- [x] Pilot on 3–5 datasets against Hydra and both controls. **Go/no-go** (revised
      2026-11-04, open question 10): mean difference ≥ −0.5 pts against both Hydra and
      `control_sym`, otherwise the study closes with a negative-result report.
      Do not use BasicMotions, GunPoint or Epilepsy as pilot datasets: every arm already
      scores 100% on them, so they cannot discriminate.
      Pilot datasets, fixed before the run (2026-11-03): Adiac and ECG5000 (UCR), NATOPS,
      EthanolConcentration and Heartbeat (UEA), × 30 resamples, `--tag pilot`.
      Run 1 (`center=True`): no-go against Hydra (−1.05). Run 2 after the diagnosis, with
      `center=False` and the revised rule: `--tag pilot2`, the same datasets and resamples.
      Run 2 result: **no-go** against Hydra (−1.37), go against `control_sym` (−0.07).
      See "M1 accuracy pilot, second run".
      Run 3, fresh datasets, fixed before the run (2026-11-04): MedicalImages,
      WordSynonyms, ScreenType (UCR), LSST and PhonemeSpectra (UEA) × 30 resamples,
      default `center="mixed"`, `--tag pilot3`. Same rule. Side check, not deciding:
      PEMS-SF, `--tag pilot3_pems`. Timing: `experiments/timing.py` on the 5 datasets
      after the accuracy runs, on an otherwise idle machine.
      Run 3 result: **go**, +0.53 against Hydra and −0.10 against `control_sym`, but
      the margin against Hydra rests on ScreenType. Timing median 12.9×. See "M1
      accuracy pilot, third run".
- [x] Freeze the design choices below and pre-register the endpoint:
      [preregistration.md](preregistration.md), frozen 2026-11-04

### M2: Full evaluation

- [x] 112 UCR + 26 equal-length UEA datasets × 30 resamples for Hanner-Hydra,
      `control_sym`, `control` and Hydra (2026-11-06, `--tag m2`; the other methods come from
      aeon's published results and SAX word counts are dropped, prereg §8)
- [x] Bayesian signed-rank test (primary): not worse than Hydra and `control_sym`
- [x] Timing endpoint on the Apple M4 (`--tag endpoint`): median ratio 11.23×. **H1
      accepted** (2026-11-07)
- [x] Secondary (2026-11-07): Wilcoxon–Holm and CD diagrams, balanced accuracy and
      macro-F1, published comparators, memory, fit times, ablations; cache misses dropped
      (counters not readable)
- [ ] C++ kernel (NEON / AVX-512), only if Numba falls short of the 4× target

### M3: Release

- [x] Make the transform aeon-compatible (2026-11-08): `HannerHydraTransformer`
      (`BaseCollectionTransformer`) and `HannerHydraClassifier` (`BaseClassifier`) in
      `hanner_hydra.estimators`, `aeon` extra
- [x] Publish results and scripts (2026-11-08): GitHub release
      [`results-v1`](https://github.com/jorgebmann/hanner-hydra/releases/tag/results-v1) with
      `hanner-hydra-results-v1.tar.gz` (29,436 result files, 7.9 MB;
      [contents](results-release.md))
- [ ] Write the paper (H1 accepted, so not a negative-results report): outline,
      bibliography and literature review in [`paper/`](../paper/) (2026-11-08)

## Hydra baseline reproduction (2026-11-03)

This is the M0 gate, run with `experiments/reproduce_hydra.py`. It runs aeon 1.6.0
`HydraClassifier(random_state=0)` on the default train/test split of each dataset and
compares it with the published resample-0 accuracy from aeon's reference results
(bake-off redux, Middlehurst, Schäfer & Bagnall 2024; timeseriesclassification.com).

| | Ours | Published (r0) |
| :--- | ---: | ---: |
| Datasets | 112 | 112 |
| Mean accuracy | 85.77 | 85.35 |
| Mean difference (ours − published) | **+0.42 pts** | |
| Median difference | ±0.00 pts | |
| Mean absolute difference | 1.10 pts | |
| Datasets with \|diff\| > 2 / > 5 pts | 19 / 4 | |

- **Gate (|mean diff| ≤ 0.5 pts): pass**, but the margin is small. The differences lean
  positive, possibly because aeon's Hydra changed between the version used for the
  reference results and 1.6.0. The largest single-dataset differences are Symbols +8.3,
  PigAirwayPressure +6.7, SmoothSubspace −6.0, Rock +6.0 and BeetleFly −5.0. Spreads like
  these are expected from one random kernel draw on one split. The 30-resample harness
  will repeat this check with the Bayesian equivalence test.
- **UCR subset.** The Hydra paper (arXiv v1, §4.1)
  evaluates "on the same 30 resamples of 112 datasets from the UCR archive" as the
  bake-off studies. These are the same 112 dataset names as aeon's reference results. The
  paper's per-dataset results are in `angus924/hydra`, `results/results_ucr112_hydra.csv`.
- **Two reference levels.** Over 30 resamples, aeon's reference averages 87.01 and the
  Hydra paper 87.35, so aeon is 0.33 pts lower. Our Hydra baseline is aeon's
  implementation, and this study compares against our own Hydra runs under the same protocol,
  so this offset does not affect the endpoint. When citing published Hydra numbers,
  state which reference they come from.
- **No multivariate references.** The Hydra paper has no UEA results, and the repo marks
  `hydra_multivariate.py` as experimental. The 30 UEA datasets need our own Hydra runs,
  and our multivariate setting, aeon's `HydraClassifier` with its multivariate
  handling, should be recorded as the baseline definition.
- The total run time was about 51 min, with all threads, on the Apple M4.

## Resampling harness (2026-11-03)

- **Runner** (`experiments/benchmark.py`). Resample 0 is the default split. Resample
  `i > 0` pools train and test and redraws a stratified split with the original class
  counts (aeon `stratified_resample_data`, seed `i`). The code is identical to
  tsml-eval 0.6.0 apart from an input check. Every arm is seeded with the resample id
  and runs on the same split. One job is one (dataset, resample) pair with all arms, in a
  single-threaded worker process. Rows go to `results/benchmark/<tag>/results.csv`
  (one per dataset, resample and arm), and interrupted runs resume. The recorded times
  come from parallel workers and are not the timing endpoint.
- **Analysis** (`experiments/analyze.py`). It uses only (dataset, resample) pairs that
  every arm completed and averages them per dataset, in points. The candidate is compared
  with each other arm by the Bayesian signed-rank test (`hanner_hydra.stats`), with ROPE
  ±0.5 and prior strength 0.5 on a pseudo-observation at 0, as in `baycomp`. The test
  reproduces `baycomp` 1.0.3 to within Monte Carlo error. `P(left/rope/right)` is the
  fraction of posterior draws in which that region is the most probable, and
  `P(not worse) = P(rope) + P(right)`.
- **Resampled splits differ from the published ones.** On 16 small UCR datasets × 6
  resamples, our Hydra runs differ from aeon's published per-resample accuracies by a
  mean |diff| of 0.78 pts on resample 0, but by 2.9–5.1 pts on resamples 1–5. The mean
  difference over all 96 runs is +0.32 pts. Resample 0 is the same split, so the larger
  gap means the published resamples 1–29 are different splits. They most likely come
  from the fixed resample files of the earlier bake-offs, generated by the Java tsml
  code, which are not distributed with aeon. Consequences:
  - The endpoint is unaffected: every arm, including Hydra, runs on our splits.
  - Comparisons with published results are only valid on per-dataset means over 30
    resamples, not paired per resample (`analyze.py --published` does this).
  - Recreating the Java resamples is possible but not planned (open question 9).
- **Cost.** In the smoke run (ArrowHead, Beef, BasicMotions × 3 resamples), the summed
  fit+predict times were Hanner 1.7 s, `control_sym` 3.0 s, `control` 3.0 s and Hydra
  7.8 s. A single Hydra pass over UCR112 took 51 min with all threads. A full 4-arm,
  30-resample UCR112 run therefore needs on the order of a day on the M4 (rough
  estimate, not measured).

## M1 accuracy pilot, first run (2026-11-03)

`experiments/benchmark.py ... --resamples 30 --tag pilot` (14 min, 8 workers, default
settings: `features="max"`, `atom_budget=256`, centering on). Mean accuracy over 30
resamples, in percent:

| Dataset | Hanner | `control_sym` | `control` | Hydra | Hanner − Hydra |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Adiac | 81.23 | 81.71 | 81.66 | 81.15 | +0.08 |
| ECG5000 | 94.60 | 94.63 | 94.59 | 94.73 | −0.13 |
| NATOPS | 83.26 | 82.96 | 83.20 | 85.28 | −2.02 |
| EthanolConcentration | 51.22 | 51.13 | 51.62 | 53.61 | −2.40 |
| Heartbeat | 76.03 | 75.89 | 75.72 | 76.83 | −0.80 |
| **Mean** | **77.27** | **77.26** | **77.36** | **78.32** | **−1.05** |

- Against `control_sym`: diff +0.00, P(not worse) 1.000. Against `control`: −0.09,
  P(not worse) 1.000. Against Hydra: −1.05, W/T/L 1/0/4, P(left/rope/right)
  0.698/0.302/0.000, so **not worse is not shown**. With 5 datasets the posterior is
  dominated by the prior, but the direction is clear.
- **The gap to Hydra is not caused by the cotree structure.** Both ternary controls sit
  exactly where Hanner does. The gap comes from the shared ternary transform, and it is
  almost entirely on the three multivariate datasets. On the two univariate datasets all
  arms are within 0.5 pts.
- Candidate causes, all shared by the three ternary arms: multivariate channel handling
  (Q5), patch centering (Q3), ternary instead of Gaussian weights, feature-budget
  matching (Q2), and float64 versus Hydra's float32.
- Median fit+predict time per resample (parallel workers, not the timing endpoint):
  Hanner is 5–17× faster than Hydra (EthanolConcentration 6.2 s vs 51.8 s, ECG5000
  3.7 s vs 62.2 s) and about 3× faster than the controls.
- Per the M1 rule this run is a **no-go against Hydra as the design stands**. The design
  is not frozen yet (open questions 2–6), so the next step is a diagnosis of the
  multivariate gap, not closing the WP. To avoid tuning on the pilot, the diagnosis
  should use other datasets, followed by one pre-declared re-run of this pilot.

## Diagnosis of the multivariate gap (2026-11-04)

**Hypothesis.** Hydra spends its 256 atoms per (dilation, input) on 32 groups of 8
kernels, each with its own random channel subset. Hanner groups have 18–32 vertices, so
the same budget gives about 11 groups (`max_vertices=18`: 15) and therefore about 11
channel subsets. This would cost accuracy on multivariate data only, as in the pilot.

**Other parity check.** aeon's `_SparseScaler` uses `torch.std`, which is unbiased
(ddof=1). Ours used ddof=0. Fixed to ddof=1 before this run (decision log).

**Design, fixed before the run.** Development datasets: ArticularyWordRecognition, Cricket,
ERing, FingerMovements, HandMovementDirection, Handwriting, Libras, RacketSports,
SelfRegulationSCP1, UWaveGestureLibrary (UEA, equal length, none in the pilot). Resamples
0–9, `--tag diag_mv`. Arms:

| Arm | Tests |
| :--- | :--- |
| `hydra` | baseline |
| `hydra:n_kernels=24,n_groups=22` | Hydra with Hanner's group count (11 groups of 24 per input) |
| `hanner`, `control_sym` | defaults |
| `hanner:max_vertices=18` | more, smaller groups (15 per block) at the same budget |
| `hanner:atom_budget=1024` | about 42 groups per block, 4× the features (diagnostic only) |
| `hanner:features=hydra` | hard-min instead of hard-max counts |
| `hanner:center=False` | no centering. Hydra's zero-mean weights act like centering, so this should not help. |

If the group-count hypothesis holds, Hydra with 11 groups drops to Hanner's level, and
Hanner gains with more groups.

**Result** (3.9 min). Mean accuracy over the 10 datasets × 10 resamples:

| Arm | Mean | vs `hanner` |
| :--- | ---: | ---: |
| `hydra` | 78.37 | +0.63 |
| `hydra:n_kernels=24,n_groups=22` | 78.26 | +0.51 |
| `hanner` | 77.75 | — |
| `control_sym` | 77.49 | −0.26 |
| `hanner:max_vertices=18` | 77.36 | −0.38 |
| `hanner:atom_budget=1024` | 78.87 | +1.12 |
| `hanner:features=hydra` | 77.75 | 0.00 (identical on every dataset) |
| `hanner:center=False` | 78.67 | +0.92 (better on 8 of 10) |

- **The group-count hypothesis is mostly rejected.** Reducing Hydra to 11 groups costs
  only 0.11 pts, and more, smaller Hanner groups (`max_vertices=18`) are worse, not
  better. A 4× atom budget helps, but at 4× the features, so it is not a fair arm.
- **Centering is the main cause.** Without centering, Hanner gains 0.92 pts and is
  0.30 pts above Hydra (P(not worse) 0.903), and 0.41 above Hydra with 11 groups
  (P(not worse) 0.974). Zero-mean Gaussian weights make Hydra's responses
  shift-invariant, so the expected effect was the opposite. The likely reason:
  centering a ternary patch is the same as using the non-ternary kernel `v - mean(v)`, so
  the level information that `v` keeps is lost, while Hydra keeps the difference input
  for slopes.
- `features="hydra"` reproduces `features="max"` exactly, as the antipode argument
  predicts.
- These datasets were used to find the effect. Next: run both controls without
  centering on the same datasets (a fair control comparison), and check centering on
  a fresh univariate development set.

**Follow-up run, fixed before the run.** (a) `--tag diag_mv` again, with the arms
`control_sym:center=False` and `control:center=False` added. (b) A univariate development
set, `--tag diag_uv`, resamples 0–9: Beef, Car, Computers, DistalPhalanxOutlineCorrect,
Earthquakes, FiftyWords, Herring, InsectWingbeatSound, Lightning7, Strawberry,
SwedishLeaf, Worms (UCR, none in the pilot). Arms: `hydra`, `hanner`, `control_sym`,
`hanner:center=False`, `control_sym:center=False`, `control:center=False`.

**Follow-up result** (7 min). Means over datasets × 10 resamples:

| Comparison | Multivariate (10) | Univariate (12) | All 22: diff, P(not worse) |
| :--- | ---: | ---: | ---: |
| `hanner` − `hydra` | −0.63 | −0.02 | −0.30, 0.541 |
| `hanner` − `control_sym` | +0.26 | −0.12 | +0.05, 0.998 |
| `hanner:center=False` − `hydra` | +0.30 | −0.18 | +0.04, 0.756 |
| `hanner:center=False` − `control_sym:center=False` | +0.55 | +0.37 | +0.45, 1.000 |
| `hanner:center=False` − `control:center=False` | +0.33 | +0.42 | +0.38, 1.000 |
| `hanner:center=False` − `hanner` | +0.92 | −0.15 | — |

- On univariate data, centering helps slightly (+0.15). On multivariate data, it costs
  0.92 pts. Without centering, Hanner ties Hydra over the 22 datasets (+0.04).
- **Without centering, the cotree structure starts to pay off.** Hanner beats both
  uncentered controls on both sets: +0.45 pts against `control_sym` (13 wins, 7 losses)
  and +0.38 against `control`. With centering, Hanner and `control_sym` are equivalent
  (+0.05).
- 22 datasets are still too few for P(not worse) ≥ 0.95 against Hydra at a mean
  difference of about zero. The pilot rule "must meet H1" cannot be met with 5 datasets
  unless the effect is large. The pilot rule needs a workable form (open question 10).

## M1 accuracy pilot, second run (2026-11-04)

`--tag pilot2`: the same 5 datasets × 30 resamples, with `center=False` (the new default)
and the revised rule (go if the mean difference is ≥ −0.5 pts against Hydra and
`control_sym`). 13.6 min. Mean accuracy in percent:

| Dataset | Hanner run 1 (centered) | Hanner run 2 | `control_sym` | `control` | Hydra | Run 2 − Hydra |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Adiac | 81.23 | 80.73 | 80.65 | 80.04 | 81.15 | −0.42 |
| ECG5000 | 94.60 | 94.49 | 94.53 | 94.52 | 94.73 | −0.24 |
| NATOPS | 83.26 | 84.70 | 84.59 | 84.44 | 85.28 | −0.57 |
| EthanolConcentration | 51.22 | 49.38 | 49.28 | 49.35 | 53.61 | −4.23 |
| Heartbeat | 76.03 | 75.46 | 76.05 | 75.45 | 76.83 | −1.37 |
| **Mean** | **77.27** | **76.95** | **77.02** | **76.76** | **78.32** | **−1.37** |

- **No-go against Hydra** under the rule fixed before the run: −1.37 pts, worse on all 5
  datasets (P(not worse) 0.298, reported only). **Go against `control_sym`** (−0.07) and
  `control` (+0.19).
- Turning off centering helped NATOPS (+1.44) but hurt EthanolConcentration (−1.84),
  Heartbeat (−0.57) and both univariate datasets. The development-set result (+0.92 on
  multivariate data) did not carry over to the pilot. EthanolConcentration alone is
  3.1 of the 6.8 pts summed gap (run 2).
- The +0.45 advantage of Hanner over `control_sym` on the development datasets does not
  show here either: Hanner and the controls are equivalent on the pilot in both runs.
- Hydra's accuracies are identical to run 1, so the harness is deterministic.
- The pilot datasets have now been used twice and are not used for further design
  decisions. Next (decided 2026-11-04): diagnose the long-series / baseline-drift case
  (EthanolConcentration-like data) on development datasets only. Then decide between
  a clean fix and an equal-time hypothesis (2–4× feature budget) tested on fresh pilot
  datasets.

## Diagnosis of the long-series / baseline-drift case (2026-11-04)

Development datasets only, none used before: Ham, Meat, OliveOil, Wine (spectra, like
EthanolConcentration), Rock, InlineSkate, CinCECGTorso, Haptics, EOGVerticalSignal,
Lightning2, SemgHandGenderCh2 (long univariate series) and SelfRegulationSCP2
(multivariate, T = 1152).

**New options** (reference and Numba, bit-identical, `tests/test_numba.py`):
`center="raw"` / `"diff"` centers one input only. `increments=True` evaluates the groups
on the tap increments `p[j+1] - p[j]`; this equals kernels `D^T v`: zero-sum, integer
entries in {−2..2}, still multiplier-free, and the exact support function of a linear
image of the Hanner polytope. `diff_share` moves atoms between the series and its
difference at a fixed total.

**Mechanism probe** (`experiments/probe_drift.py`, 40 training cases, resample 0).
`drift` is the share of patch energy in the patch mean at dilation 1. `top1` is the mean
share of anchors won by a group's most frequent atom, and `eff` is the effective number
of winners, exp(entropy), as a fraction of the group size. Raw-series input:

| Dataset | drift | Hydra eff | Hanner eff (top1) | centered eff | increments eff |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Ham | 0.80 | 0.81 | 0.55 (0.22) | 0.55 | 0.73 |
| Meat | 0.99 | 0.66 | 0.27 (0.40) | 0.37 | 0.59 |
| OliveOil | 0.95 | 0.71 | 0.43 (0.25) | 0.42 | 0.67 |
| Wine | 0.92 | 0.71 | 0.37 (0.28) | 0.41 | 0.64 |
| Rock | 1.00 | 0.66 | 0.11 (0.65) | 0.37 | 0.70 |
| InlineSkate | 1.00 | 0.67 | 0.25 (0.38) | 0.40 | 0.68 |
| CinCECGTorso | 0.97 | 0.79 | 0.41 (0.31) | 0.53 | 0.80 |
| Haptics | 0.98 | 0.64 | 0.34 (0.27) | 0.35 | 0.57 |
| EOGVerticalSignal | 1.00 | 0.69 | 0.25 (0.48) | 0.41 | 0.74 |
| Lightning2 | 0.76 | 0.86 | 0.36 (0.36) | 0.64 | 0.83 |
| SemgHandGenderCh2 | 0.81 | 0.94 | 0.17 (0.50) | 0.80 | 0.84 |
| SelfRegulationSCP2 | 0.99 | 0.81 | 0.17 (0.54) | 0.59 | 0.77 |

- **Winner collapse on the raw input.** Uncentered ternary groups put most anchors on
  one or a few vertices when the patches are level-dominated (Rock: one vertex wins 65%
  of the anchors). `control_sym` behaves the same (not shown: eff 0.13–0.55). The hard
  counts of such a group carry little information.
- **Centering fixes this only partly.** The centered kernel `v - mean(v)` keeps the
  low-frequency content of `v`, so on smooth patches a few ramp-like vertices still
  dominate.
- **Increments restore Hydra-level spread** on every dataset. Hydra's eff is a fraction
  of 8 kernels, ours of 18–32, so matching it is the harder case.
- **The difference input is not the problem:** eff 0.6–0.9 for every ternary arm, in
  line with Hydra.

**Accuracy run, fixed before the run.** The 12 datasets above, resamples 0–9,
`--tag diag_drift`. Arms: `hydra`, `hydra:n_groups=128` (2× Hydra budget), `hanner`,
`control_sym`, `hanner:center=True`, `hanner:increments=True`,
`control_sym:increments=True`, `control:increments=True`, `hanner:diff_share=0.75`,
`hanner:atom_budget=512`, `hanner:atom_budget=1024`,
`hanner:increments=True,atom_budget=512`.

**Accuracy result** (155 min; 835 of the 1,205 CPU-minutes went to `hydra:n_groups=128`).
Mean over the 12 datasets × 10 resamples:

| Arm | Mean | vs `hanner` | vs `hydra` |
| :--- | ---: | ---: | ---: |
| `hydra` | 78.45 | +0.35 | — |
| `hydra:n_groups=128` (2× budget) | 78.72 | +0.61 | +0.27 |
| `hanner` (uncentered default) | 78.10 | — | −0.35 |
| `control_sym` | 78.46 | +0.36 | +0.01 |
| `hanner:center=True` | 78.71 | +0.61 | +0.26 |
| `hanner:increments=True` | 77.52 | −0.58 | −0.93 |
| `control_sym:increments=True` | 77.63 | −0.47 | −0.82 |
| `control:increments=True` | 77.41 | −0.69 | −1.04 |
| `hanner:increments=True,atom_budget=512` | 77.42 | −0.68 | −1.03 |
| `hanner:diff_share=0.75` | 76.96 | −1.15 | −1.49 |
| `hanner:atom_budget=512` (2×) | 78.14 | +0.04 | −0.31 |
| `hanner:atom_budget=1024` (4×) | 78.40 | +0.30 | −0.05 |

Summed fit+predict CPU time over all 120 jobs: `hanner` 6.4 min, `atom_budget=1024`
22.2 min, `control_sym` 24.3 min, `hydra` 214 min, `hydra:n_groups=128` 835 min.

- **Restoring winner spread does not restore accuracy.** Increments give Hydra-level
  spread (probe) but cost 0.58 pts, for Hanner and both controls alike. The losses are
  largest where low-frequency shape matters (SemgHandGenderCh2 −6.0, Lightning2 −2.6,
  Wine −1.9), and the gains where they are small (Ham +3.2, CinCECGTorso +1.1).
  Increments are a high-pass operation and discard exactly the information these
  datasets need. The collapse in the hard counts is real, but the soft-max values of a
  collapsed group still carry the level and shape signal. **Zero-sum integer vertices are
  rejected as a fix.**
- **Difference-channel weighting is rejected.** Moving atoms to the difference input
  costs 1.15 pts. The raw series carries the information on these datasets.
- **Centering is the lever, and its sign depends on the data.** It gains 0.61 pts here
  (spectra and long smooth series, consistent with EthanolConcentration in pilot run 2),
  gains 0.15 on the univariate development set, and loses 0.92 on the multivariate one.
  Over all 34 development datasets both settings sit at about −0.10 pts against Hydra.
  No single global choice wins everywhere, so the EthanolConcentration gap in pilot run 2
  is mostly the centering choice, not a separate long-series failure.
- **Budget scaling helps both methods by a similar small amount.** Hanner at 4× atoms:
  +0.30; Hydra at 2× groups: +0.27. Hanner at 4× ties default Hydra (−0.05) at about a
  tenth of its fit+predict CPU time (parallel workers; not the timing endpoint).
- **Cotree structure:** on this set `control_sym` is 0.36 pts *ahead* of Hanner. Over
  the 34 development datasets, Hanner and its matched controls are equivalent within
  noise. The +0.45 on the first two development sets did not replicate.
- **Candidate clean fix, untested:** use both settings in every block, with half of the
  groups centered and half not, at the same atom budget, and let the ridge pick per
  dataset. This removes a dataset-dependent design choice instead of tuning it.

## Mixed centering on all development datasets (2026-11-04)

**Option**: `center="mixed"`. In every block the odd-numbered groups are centered and the
even ones are not. The dictionaries are the same as for the default, and the atom budget
is unchanged (reference and Numba, bit-identical).

**Design, fixed before the run.** All 34 development datasets, resamples 0–9:

- the multivariate set (10, `--tag diag_mix_mv`) and the univariate set (12,
  `--tag diag_mix_uv`), new tags because the recorded transform defaults changed. Arms:
  `hydra`, `hanner`, `hanner:center=True`, `hanner:center=mixed`,
  `control_sym:center=mixed`, `control:center=mixed`.
- the drift set (12, `--tag diag_drift`, resumed), adding `hanner:center=mixed`,
  `control_sym:center=mixed` and `control:center=mixed`.

**Decision rule for the next step**:
- Mixed centering becomes the candidate design if both conditions hold over the 34
  datasets:
  - its mean difference to Hydra is ≥ −0.1 pts (the better of the two global settings
    is −0.10);
  - it is not worse than either fixed setting on any of the three sets by more than
    0.5 pts.
- Then it goes to a fresh pilot. Otherwise the recommendation is to close the study with a
  negative-result report.

**Result** (14 min). Mean accuracy over datasets × 10 resamples:

| Arm | Multivariate (10) | Univariate (12) | Drift (12) | All 34 |
| :--- | ---: | ---: | ---: | ---: |
| `hydra` | 78.37 | 81.26 | 78.45 | 79.42 |
| `hanner` (uncentered) | 78.67 | 81.08 | 78.10 | 79.32 |
| `hanner:center=True` | 77.75 | 81.24 | 78.71 | 79.32 |
| `hanner:center=mixed` | 78.31 | 81.37 | 78.69 | **79.52** |
| `control_sym:center=mixed` | 77.95 | 81.34 | 78.74 | 79.43 |
| `control:center=mixed` | 78.63 | 80.98 | 78.53 | 79.42 |

- **The decision rule is met.** Mixed versus Hydra: +0.11 pts over 34 datasets
  (≥ −0.1), with 15 wins and 18 losses, P(not worse) 0.875. Mixed versus the fixed
  settings, per set: against uncentered −0.36 / +0.29 / +0.59, against centered
  +0.56 / +0.13 / −0.01, so it is never more than 0.5 pts worse. Mixed centering
  becomes the candidate design for a fresh pilot.
- Mixed is the most robust setting, but its gain is small: +0.20 over uncentered and
  +0.21 over centered Hanner. These 34 datasets were used to choose it, so the +0.11
  against Hydra is optimistic.
- **Cotree structure:** under mixed centering, Hanner is +0.10 pts ahead of both
  controls (P(not worse) 1.000 and 0.998, P(right) 0.004 and 0.024). That makes it
  equivalent to them, not better. The accuracy part of H1 does not need it to be better,
  but the WP's scientific claim, that the cotree structure matters, has no support so
  far beyond speed: the Hanner kernel is about 3–4× faster than the equally optimized
  controls.
- Hydra's and uncentered Hanner's numbers on the multivariate set are identical to the
  `diag_mv` run, so the harness stays deterministic.

## M1 accuracy pilot, third run (2026-11-04)

**Design** (fixed before the run, see M1 and the decision log): MedicalImages,
WordSynonyms, ScreenType (UCR), LSST and PhonemeSpectra (UEA) × 30 resamples, default
`center="mixed"`, `--tag pilot3`. None of these datasets was used during development. Rule:
mean difference ≥ −0.5 pts against both Hydra and `control_sym`.

**Run note.** The first attempt hung after 128 of 150 jobs: all pool workers were gone and
the main process waited forever. The cause was worker recycling (`max_tasks_per_child`) in
Python 3.12's `ProcessPoolExecutor`. The hung run was stopped, recycling was removed from
`experiments/benchmark.py`, and the remaining 22 PhonemeSpectra jobs were resumed under the
same tag and config. The results do not depend on the worker layout (each job is seeded by
its resample id).

**Result.** Mean accuracy over 30 resamples, in points; Δ is Hanner minus the arm, with the
paired per-resample wins / losses of Hanner:

| Dataset | Hanner | `control_sym` | `control` | Hydra | Δ Hydra (W/L) | Δ `control_sym` (W/L) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSST | 62.20 | 61.47 | 61.38 | 60.26 | +1.94 (28/2) | +0.73 (22/7) |
| MedicalImages | 75.75 | 76.08 | 76.16 | 76.54 | −0.79 (8/22) | −0.33 (10/20) |
| PhonemeSpectra | 30.63 | 30.88 | 30.53 | 33.08 | −2.46 (0/30) | −0.25 (10/20) |
| ScreenType | 68.91 | 68.76 | 69.15 | 62.76 | +6.14 (29/1) | +0.15 (13/13) |
| WordSynonyms | 72.41 | 73.20 | 72.93 | 74.58 | −2.17 (1/29) | −0.79 (8/20) |
| **Mean** | **61.98** | **62.08** | **62.03** | **61.45** | **+0.53** | **−0.10** |

- **Go.** Against Hydra +0.53 pts, against `control_sym` −0.10 pts (against `control`
  −0.05). Both are above −0.5, so the pre-declared rule is met.
- Bayesian signed-rank test, reported only (5 datasets, prior-dominated): P(not worse)
  0.565 against Hydra (P(left) 0.435, P(right) 0.540), 0.950 against `control_sym` and
  0.995 against `control`.
- **The go against Hydra rests on ScreenType.** Hanner is behind Hydra on three of five
  datasets (median Δ −0.79), and without ScreenType the mean would be −0.87. Each
  per-dataset difference is consistent across resamples (standard errors 0.1–0.5 pts),
  so this is dataset heterogeneity, not resampling noise: the ternary arms gain on
  ScreenType and LSST and lose on PhonemeSpectra and WordSynonyms, all three of them
  together. Leave-one-out against `control_sym` stays between −0.31 and +0.08.
- **Cotree structure:** again equivalent to the controls, not better (−0.10 and −0.05).
  As on the development datasets, the case for the cotree structure is speed, not accuracy.
- The M2 decision (H1 over 112 UCR datasets, P(not worse) ≥ 0.95 against Hydra) is
  therefore open: the development mean (+0.11 over 34 datasets) and this pilot both put
  Hanner close to Hydra, with a dataset-level spread of several points either way.

**Side check, PEMS-SF** (C = 963, `--tag pilot3_pems`, 30 resamples, does not decide):
Hanner 82.76, `control_sym` 82.56, `control` 83.58, Hydra 85.84. Hanner is −3.08 pts behind
Hydra, and `control_sym` is similar (−3.28). This is consistent with the channel-subsampling
bottleneck expected from about 11 groups of at most 8 channels per block, and it affects the
ternary design generally, not the cotree structure. Candidate for a fix before M2 (open
question 11).

**Timing** (`experiments/timing.py`, one thread, Apple M4, idle machine, the median of 7
`transform(X_test)` calls):

| Dataset | Hydra | Hanner | `control_sym` | `control` | Hydra / Hanner |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MedicalImages | 1.73 s | 0.10 s | 0.39 s | 0.39 s | 16.7× |
| WordSynonyms | 4.05 s | 0.32 s | 1.16 s | 1.18 s | 12.9× |
| ScreenType | 6.86 s | 0.54 s | 2.00 s | 2.01 s | 12.8× |
| LSST | 1.68 s | 0.13 s | 0.48 s | 0.50 s | 13.5× |
| PhonemeSpectra | 17.42 s | 1.59 s | 5.64 s | 5.72 s | 11.0× |

- Median ratio **12.9×**, every dataset above the 4× target. Hanner is about 3.6× faster
  than the equally optimized controls.
- Higher than the 8.9× of the early six-dataset check, whose datasets were longer or had
  more channels; not a like-for-like comparison. The endpoint still needs the controlled
  timing harness.

## Diagnosis of high-channel datasets (2026-11-04)

**Question** (open question 11): on PEMS-SF (C = 963) Hanner and `control_sym` are about
3 pts behind Hydra. Hypothesis: channel coverage. Per (dilation, input), Hydra has 32 groups
of `clip(C // 2, 2, 8)` channels (256 channel draws for C ≥ 16); the ternary arms have
about 11 groups of the same size (about 88 draws).

**Levers.** More groups at the same budget is limited: a cotree on 9 taps has at least
18 vertices, so even `max_vertices=18` gives only about 15 groups, and it cost 0.38 pts on
the low-channel development set ("Diagnosis of the multivariate gap"). The other lever
is the channel cap: new option `max_channels` (default 8, as in Hydra and aeon's
`max_num_channels`). `"auto"` raises the cap per (dilation, input) to
`max(8, ceil(budget / G))`, i.e. at least Hydra's 256 channel draws (about 24 channels
per group at G = 11). Only data with C ≥ 18 is affected, the random stream is unchanged,
so the 112 UCR datasets cannot be affected. *Correction (2026-11-04): this sentence
first called the 112 UCR datasets "the primary endpoint". The endpoint includes the UEA
datasets, of which 7 have C ≥ 18.*

**Design, fixed before the run.**

- Development set: the four equal-length UEA datasets with C ≥ 18 that are already used:
  FingerMovements (28), NATOPS (24), Heartbeat (61), PEMS-SF (963). Resamples 0–9,
  `--tag diag_hc`.
- Confirmation set, untouched so far: DuckDuckGeese (1345), FaceDetection (144),
  MotorImagery (64). Resamples 0–9, `--tag diag_hc_confirm`, only if the development
  rule passes, and only with the arm that passed.
- Arms: `hydra`, `hydra:n_kernels=24,n_groups=22` (Hydra with Hanner's group count: does
  Hydra lose the same on high-C data?), `hanner`, `control_sym`, `control`,
  `hanner:max_channels=16`, `hanner:max_channels=32` (dose-response, reported only),
  `hanner:max_vertices=18` (more groups), and `max_channels=auto` for all three ternary
  families.

**Decision rule.**

- Development: `max_channels="auto"` goes to confirmation if `hanner:max_channels=auto`
  gains ≥ +0.5 pts over `hanner` on the mean of the 4 datasets and loses no more than
  1 pt on any of them. The fixed caps are reported, not chosen from.
- Confirmation: the mean gain over the 3 fresh datasets is ≥ 0. Then `"auto"` becomes
  the default for all ternary arms (they are compared under the same setting).
  Otherwise the default stays 8, and the high-channel gap is a known limitation for M2.

**Result** (`diag_hc`, 1.5 min). Mean accuracy over resamples 0–9:

| Arm | FingerMovements | Heartbeat | NATOPS | PEMS-SF | Mean | vs `hanner` |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hydra` | 54.20 | 76.49 | 85.11 | 84.68 | 75.12 | +1.59 |
| `hydra:n_kernels=24,n_groups=22` | 52.60 | 76.49 | 84.72 | 85.32 | 74.78 | +1.25 |
| `hanner` | 53.10 | 75.17 | 83.61 | 82.25 | 73.53 | — |
| `control_sym` | 52.20 | 75.66 | 83.94 | 81.97 | 73.44 | −0.09 |
| `control` | 55.90 | 75.22 | 84.39 | 82.49 | 74.50 | +0.96 |
| `hanner:max_channels=16` | 52.50 | 75.37 | 83.83 | 83.76 | 73.86 | +0.33 |
| `hanner:max_channels=32` | 52.50 | 74.73 | 83.83 | 84.97 | 74.01 | +0.48 |
| `hanner:max_vertices=18` | 51.80 | 74.44 | 84.72 | 83.64 | 73.65 | +0.12 |
| `hanner:max_channels=auto` | 52.50 | 74.24 | 83.83 | 83.58 | 73.54 | +0.01 |
| `control_sym:max_channels=auto` | 53.20 | 74.83 | 82.78 | 82.95 | 73.44 | −0.10 |
| `control:max_channels=auto` | 52.10 | 74.54 | 82.89 | 83.93 | 73.36 | −0.17 |

- **The development rule fails.** `max_channels="auto"` gains +0.01 pts on the mean
  (needed ≥ +0.5): +1.33 on PEMS-SF, −0.60 on FingerMovements, −0.93 on Heartbeat and
  +0.22 on NATOPS (where the three raised caps coincide, since `C // 2 = 12`). No
  confirmation run; **the default stays `max_channels=8`**.
- **Channel coverage does not explain Hydra's lead.** Hydra with Hanner's group count
  (11 groups of 24 per input, the same ~88 channel draws as Hanner) loses nothing on
  PEMS-SF (+0.64 ± 1.31) and only 0.34 pts on the mean. So Hydra is ahead at equal
  coverage; the gap comes from the dictionaries or features, not from how many channels
  the groups see.
- More channels do help the ternary arms on PEMS-SF alone: +2.72 ± 1.11 with a cap of
  32 (8 of 10 resamples), close to Hydra. They do not help on the other three datasets,
  where the differences are within noise (FingerMovements is near chance with 100 test
  cases, standard error 1.6 pts). A rule like "raise the cap for very large C" would be
  chosen after seeing one dataset, so it is not adopted. It is a candidate ablation for
  M2.
- More, smaller groups (`max_vertices=18`): +0.12, no clear effect.
- `control` is +0.96 ahead of `hanner` here, almost all of it on FingerMovements
  (+2.8, within about 2 standard errors on a near-chance dataset).
- `max_channels` stays in the code (default 8, bit-identical to before) for that ablation.

## M2 accuracy runs (2026-11-06)

**Design:** pre-registered ([preregistration.md](preregistration.md) §3–§5). 112 UCR and 26 equal-length UEA datasets × 30 resamples × 4 arms
(`hanner`, `control_sym`, `control`, Hydra), `--tag m2`, frozen defaults. Every arm
completed every resample on all 138 datasets (16,560 prediction files), with no failures and
no dataset excluded.

**Run note.** The run was interrupted twice and resumed under the same tag and config.
- At 8 and then 6 workers the Mac ran out of memory, once at about 10.5 h and once on
  StarLightCurves. Since worker recycling was removed (pilot run 3), pool workers keep
  their memory for the whole invocation.
- The remainder ran under a scratchpad chain with one `benchmark.py` invocation per
  dataset, so each dataset got a fresh pool.
- Large datasets got fewer workers: StarLightCurves, UWaveGestureLibraryAll, MotorImagery,
  DuckDuckGeese, PEMS-SF, FaceDetection and EthanolConcentration got 2, EigenWorms got 1.
- A memory watchdog stopped the run once when free memory fell below 15 %. The resamples
  that were running then were simply rerun.
- None of this affects the results: each job is seeded by its resample id, and the harness
  code was not changed.

**Primary result** (`analyze.py --tag m2`; Bayesian signed-rank test on the 138 dataset
means, ROPE ±0.5 pts; Δ is Hanner minus the arm; W/T/L counts datasets):

| Hanner-Hydra vs | Mean (pts) | Δ | W/T/L | P(left / rope / right) | P(not worse) | |
| :--- | :--- | ---: | :--- | :--- | ---: | :--- |
| Hydra | 84.37 vs 84.43 | −0.06 | 47/5/86 | 0.012 / 0.988 / 0.000 | 0.988 | **not worse** |
| `control_sym` | 84.37 vs 84.33 | +0.04 | 64/7/67 | 0.000 / 1.000 / 0.000 | 1.000 | **not worse** |
| `control` | 84.37 vs 84.26 | +0.11 | 72/5/61 | 0.000 / 1.000 / 0.000 | 1.000 | not worse |

- **Decision conditions 1 and 2 hold** (prereg §5). Both comparisons are also practically
  equivalent (P(rope) ≥ 0.95). H1 itself waits for condition 3, the timing endpoint.
- **Equivalent, not better.** Hanner loses more datasets to Hydra than it wins (47/86), with
  a median Δ of −0.10. The differences are small: 18 of 138 datasets have |Δ| > 2 pts,
  ranging from −5.43 (FreezerSmallTrain) to +6.79 (SemgHandMovementCh2).
- **The cotree structure is again equivalent to the matched controls, not better** (+0.04,
  +0.11). This is as in all M1 runs. The case for the cotree structure is speed.
- W/T/L counts are from the analysis after deviation D1 (prereg §11). Before D1 they varied
  by one or two datasets between runs. The means and probabilities are unchanged.

**Secondary analyses** (prereg §6, not deciding):

| Subset | n | Δ Hydra | P(not worse) | Δ `control_sym` | P(not worse) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| UCR (`ucr112`) | 112 | +0.06 (40/5/67) | 0.999 | +0.00 | 1.000 |
| UEA (`uea26`) | 26 | −0.54 (7/0/19) | **0.220** | +0.23 | 1.000 |
| Untouched (`fresh`) | 93 | +0.01 (34/5/54) | 1.000 | +0.04 | 1.000 |

- `control_sym` vs Hydra over all 138 datasets: −0.10 (50/6/82), P(not worse) 1.000.
- Our Hydra arm vs aeon's published Hydra results (UCR, 3,360 resamples): +0.33 pts,
  P(rope) 1.000. The baseline reproduces.
- **The UEA gap belongs to the ternary design, not the cotree structure.** `control_sym` is
  −0.77 behind Hydra on UEA as well.
  - The gap is larger with many channels: −1.19 on the 7 datasets with C ≥ 18 (PEMS-SF
    −3.08, NATOPS −1.54, Heartbeat −1.32, MotorImagery −1.03).
  - It is −0.30 on the 19 others, which include the largest single loss, AtrialFibrillation
    (C = 2, −4.89, 15 test cases).
  - This is consistent with the high-channel diagnosis, which found that channel coverage
    does not explain it. A UEA-only claim of "not worse than Hydra" cannot be made.
- On the 93 datasets that informed no design choice, the result is the same as on all
  datasets, so it is not carried by the development datasets.
- Not done yet: Wilcoxon–Holm and CD diagrams, balanced accuracy and macro-F1 (already in
  `results.csv`), the published comparators (WEASEL 2.0, MiniROCKET, MultiROCKET, HC2),
  memory, and the ablations.

## M2 timing endpoint (2026-11-07)

**Design:** pre-registered (prereg §4). `experiments/timing.py --suite ucr112 --suite
uea26 --tag endpoint` on the Apple M4, on mains power with Low Power Mode off. One thread.
Arms: Hydra, `hanner`, `control_sym` and `control`. One warm-up, then 7 interleaved
`transform(X_test)` calls per arm. 3 passes in separate processes, each with its own seeded
dataset order. All 138 datasets completed all 3 passes with no errors.

**Run note: pass 0 was repeated.**
- The first pass 0 ran from 02:40 to 20:05 on 2026-11-06, partly while the Mac was in use.
  That breaks the idle-machine condition, so the pass was repeated as the protocol requires,
  on 2026-11-07, single-process.
- The original is kept as `pass0_nonidle_20261106.csv` and its summary as
  `summary_with_nonidle_pass0.csv`.
- The two pass 0 runs agree: per-dataset time ratio new / old median 1.001 for Hydra and
  1.008 for Hanner. With the original pass 0 the endpoint was also 11.23×.
- Both daytime pass 0 runs took longer in wall time (17.4 h and 11.3 h) than the 6.4 h of
  timed work plus fits. The night passes took 6.4 h each. There were no sleep events. The
  extra time fell outside the timed calls, whose values and spreads match the night passes.
- `caffeinate -i -s` kept the Mac from sleeping during the rerun. The screen lock does not
  pause processes.

**Result** (`timing.py --tag endpoint --summary`; per dataset, the median of the three
pass medians):

| | All 138 | UCR 112 | UEA 26 |
| :--- | ---: | ---: | ---: |
| Median Hydra / Hanner ratio | **11.23×** | 11.76× | 9.38× |
| Minimum | 4.06× (DuckDuckGeese) | 6.91× | 4.06× |
| Maximum | 23.15× (StarLightCurves) | 23.15× | 21.61× |
| Median `control_sym` / Hanner | 3.79× | 3.80× | 3.62× |

- **Decision condition 3 holds**: 11.23× ≥ 4. The bootstrap 95% interval is 10.23–12.01,
  and the per-pass medians are 11.17×, 11.24× and 11.24×.
- Every dataset is above 4×. 17 are below 8×, and the three slowest are DuckDuckGeese (C =
  1,345) at 4.06×, PEMS-SF (C = 963) at 4.83× and StandWalkJump (length 2,500) at 5.86×.
- Summed over all datasets, one transform of every test set takes 29.0 min for Hydra, 2.2
  min for Hanner and 8.3 min for `control_sym` (13.1× in total).
- **Most of the speed comes from the cotree recursion.** The equally optimized ternary
  controls take 3.8× longer than Hanner. Hydra versus `control_sym`, about 3×, is the gain
  from the ternary Numba kernel without the cotree.
- Not done yet (prereg §6): peak transform memory, fit times, and cache misses through
  Apple's counters.

## H1 decision (2026-11-07)

**H1 is accepted** (prereg §5). All three conditions hold:

1. Not worse than Hydra in accuracy: P(not worse) 0.988 (Δ −0.06 pts, 138 datasets).
2. Not worse than `control_sym`: P(not worse) 1.000 (Δ +0.04 pts).
3. Median transform-time ratio Hydra / Hanner-Hydra 11.23× ≥ 4.

In the terms of prereg §5, Hanner-Hydra is *practically equivalent* to Hydra and to
`control_sym` in accuracy (P(rope) 0.988 and 1.000), and about 11× faster than Hydra.

**Caveats to state with it:**
- It is equivalent, not better. Hanner loses more datasets to Hydra than it wins (47/86).
- On UEA alone, not worse than Hydra is not shown (Δ −0.54, P 0.22). `control_sym` shares
  this gap, so it belongs to the ternary design.
- The cotree structure is not more accurate than the matched controls. Its contribution is
  the 3.8× speed-up over them.

## M2 secondary analyses (2026-11-07)

Pre-registered in prereg §6, exploratory, and not deciding. The analyses come from
`experiments/secondary.py all --tag m2 --timing-tag endpoint` and
`experiments/memory.py`. Tables and figures are in `results/benchmark/m2/secondary/` and
`results/memory/endpoint/`.

### Frequentist tests and CD diagrams

Two-sided Wilcoxon signed-rank tests run on the dataset means for every pair of the four
arms, Holm-corrected over the 6 pairs, with α = 0.05. The prereg does not fix α, so 0.05 was
chosen before running the tests.

| Pair (138 datasets) | Δ | W/T/L | p | p (Holm) |
| :--- | ---: | :--- | ---: | ---: |
| Hanner vs `control_sym` | +0.04 | 64/7/67 | 0.92 | 0.92 |
| Hanner vs `control` | +0.11 | 72/5/61 | 0.25 | 0.50 |
| Hanner vs Hydra | −0.06 | 47/5/86 | 0.054 | 0.22 |
| `control_sym` vs `control` | +0.07 | 76/7/55 | 0.029 | 0.15 |
| `control_sym` vs Hydra | −0.10 | 50/6/82 | 0.069 | 0.22 |
| `control` vs Hydra | −0.17 | 48/5/85 | 0.012 | 0.073 |

- **No pair differs significantly after Holm.** The Friedman test is significant (p 0.0002)
  because Hydra ranks first more often. Average ranks: Hydra 2.11, `control_sym` 2.53,
  Hanner 2.61, `control` 2.75.
- **UEA:** Hanner vs Hydra has p 0.036, Holm 0.14. `control_sym` vs Hydra is significant
  (Δ −0.77, Holm 0.017).
- **UCR:** `control_sym` vs `control` is significant (Δ +0.09, Holm 0.045). This is a
  negligible effect of central symmetry.
- **Untouched (`fresh`):** nothing is significant.
- **CD diagrams** (`cd_{all,ucr112,uea26,fresh}.png/.pdf`) use aeon's
  `plot_critical_difference`, the bake-off convention. Its cliques come from one-sided
  Wilcoxon tests at a fixed α/(k−1), not from the step-down Holm procedure in the table, so
  the diagrams can split pairs that the table does not. On all 138 datasets the diagram
  separates `control` from Hydra (p 0.012 < 0.05/3); Hanner, `control_sym` and Hydra form
  one clique.

### Balanced accuracy and macro-F1 (Bayesian signed-rank, ROPE 0.5)

| Metric, 138 datasets | Hanner vs Hydra | P(not worse) | Hanner vs `control_sym` | P(not worse) |
| :--- | ---: | ---: | ---: | ---: |
| Accuracy | −0.06 | 0.988 | +0.04 | 1.000 |
| Balanced accuracy | −0.16 | **0.853** | +0.03 | 1.000 |
| Macro-F1 | −0.16 | **0.788** | +0.03 | 1.000 |

- **Against Hydra, not worse is not shown on balanced accuracy or macro-F1** (inconclusive).
  Hanner is a little further behind on these metrics than on accuracy, so it loses slightly
  more on minority classes.
- Against both controls, Hanner is practically equivalent on every metric.
- On UEA, both metrics against Hydra are inconclusive (Δ −0.69 and −0.76).

### Other methods (aeon's published results, 112 UCR datasets, dataset means)

| Hanner-Hydra vs | Mean | Δ | W/T/L | P(left / rope / right) | Label |
| :--- | :--- | ---: | :--- | :--- | :--- |
| WEASEL 2.0 | 87.40 vs 87.38 | +0.01 | 58/3/51 | 0.441 / 0.080 / 0.479 | inconclusive |
| MiniROCKET | 87.40 vs 87.44 | −0.05 | 57/4/51 | 0.474 / 0.110 / 0.416 | inconclusive |
| MultiROCKET | 87.40 vs 88.14 | −0.75 | 42/5/65 | 0.991 / 0.003 / 0.007 | practically worse |
| HIVE-COTE 2.0 | 87.40 vs 89.14 | −1.75 | 30/4/78 | 1.000 / 0.000 / 0.000 | practically worse |

- "Practically worse" (P(Δ < −0.5) ≥ 0.95) is the mirror of the *better* label (prereg §5).
- Against WEASEL 2.0 and MiniROCKET, the posterior splits between the two sides of the ROPE,
  with large per-dataset differences in both directions. The results are on par, but
  equivalence is not shown.
- In the 8-method CD diagram (`cd_published_ucr112.png`), the average ranks are HC2 3.11,
  MultiROCKET 3.69, Hydra 4.35, `control_sym` 4.66, Hanner 4.92, `control` 5.02, WEASEL
  2.0 5.11 and MiniROCKET 5.14. Against Hanner, only HC2 is significant after Holm (p 2e-6).
- The published resample splits differ from ours (open question 9). Pairing on dataset
  means takes care of this, but the comparison is less tight than for our own arms.

### Memory (`experiments/memory.py --tag endpoint`)

Each (dataset, arm) pair is measured in a fresh process with one thread. The measure is the
rise in peak RSS during `transform(X_test)` after fit and a warm-up, and it includes the
output array.

| | Hydra | Hanner | `control_sym` | `control` |
| :--- | ---: | ---: | ---: | ---: |
| Median extra peak | 450 MB | 19 MB | 17 MB | 17 MB |
| Maximum | 16.8 GB (StarLightCurves) | 421 MB (Crop) | 416 MB | 353 MB |
| Median beyond the output | 435 MB | 1.1 MB | 1.1 MB | n/a |

- **Hydra needs 23.8× more transform memory** (median ratio). It needs more on all 138
  datasets; the closest cases are DuckDuckGeese (1.17×) and PEMS-SF (1.32×).
- **Hanner's peak is essentially its output**: a median of 14 MB of features (more
  features than Hydra, in float64), with about 1 MB of working memory. The cotree recursion
  needs no memory beyond the controls (ratio 0.99×).
- Hydra's 16.8 GB on StarLightCurves explains the memory failures of the parallel M2 runs.
- RSS can underestimate when the allocator reuses freed pages (3 Hanner measurements are
  below 1 MB). The procedure is the same for every arm.

### Fit times

- Transform fit (kernel sampling, from the timing passes): medians are 1 ms for Hydra, 5 ms
  for Hanner, 28 ms for `control_sym` and 38 ms for `control`. All are negligible next to
  the transform.
- Classifier fit and predict in the M2 run are indicative only, because they were measured
  in parallel workers. Medians over datasets: Hydra / Hanner 8.0× for fit and 12.2× for
  predict; `control_sym` / Hanner 3.3× and 3.6×. Fit includes the transform of the training
  set and the ridge fit.

### Cache misses: dropped

The pre-registered fallback (prereg §6) applies.
- Developer mode is off on this managed Mac, and enabling it needs administrator rights.
- `xctrace` with the *CPU Counters* template wrote 754 MB of raw counter data for a
  2-second process, but it did not produce a readable trace ("Document Missing Template
  Error").
- The private `kpc` interface needs root.
- The counters cannot be read reliably, so the metric is not reported.

### Ablations (exploratory, `--tag ablations`, Hanner only, 30 resamples)

| Ablation | n | Mean | Δ vs default Hanner | Δ vs Hydra | Note |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Default (`center="mixed"`) | 138 | 84.37 | | −0.06 | |
| `center=False` | 138 | 83.69 | −0.68 (26/8/104) | −0.74 | UCR −0.79, UEA −0.21 |
| `center=True` | 138 | 84.35 | −0.02 (60/8/70) | −0.08 | practically equivalent to the default; UEA +0.13 |
| `features="hydra"` | 138 | 84.37 | 0.00 (0/138/0) | −0.06 | identical predictions |
| `max_channels=32` | 7 | 65.78 | −0.90 (2/0/5) | −2.09 | the 7 UEA datasets with C ≥ 18 |
| `max_channels="auto"` | 7 | 66.14 | −0.53 (3/0/4) | −1.72 | the same 7 datasets |

- **Centering:** mixed and full centering are equivalent. No centering costs 0.68 pts,
  mostly on UCR. The development diagnosis had predicted the opposite for multivariate data,
  but on the full UEA set full centering is +0.13 against mixed.
- **`features="hydra"`** gives the same accuracy on all 4,140 resamples, and all 4,140
  prediction files are byte-identical to the default arm's (checked 2026-11-08). For centrally symmetric groups, the hard-min counts are
  a permutation of the hard-max counts (decision log 2026-11-03), and the scaler and ridge
  are invariant to that.
- **`max_channels`:** raising the cap makes the high-channel datasets worse, not better.
  This confirms the decision to keep 8: channel coverage is not the cause of the UEA gap.
- On the other 131 datasets, `max_channels` is identical to the default by construction
  (`tests/test_transform.py`), so those arms ran on the 7 datasets with C ≥ 18 only.

## Early timing results (2026-11-03)

These come from `experiments/timing.py`: one thread on an Apple M4, the default
train/test split, and the median of 7 `transform(X_test)` calls after a warm-up. They
compare against aeon 1.6.0 `HydraTransformer(8, 64)`. This is an early check, not the
endpoint.

**Kernel v2** (blocked, node-major, pre-padded). The v1 column is the first Numba kernel:

| Dataset | n × C × T | Hydra | Hanner | control_sym | control | Hydra / Hanner | v1 ratio |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| GunPoint | 150 × 1 × 150 | 0.55 s | 0.055 s | 0.22 s | 0.21 s | 9.9× | 4.7× |
| ArrowHead | 175 × 1 × 251 | 0.92 s | 0.12 s | 0.42 s | 0.42 s | 7.9× | 4.5× |
| FordA | 1320 × 1 × 500 | 27.4 s | 1.5 s | 5.7 s | 5.8 s | 17.9× | 6.9× |
| NATOPS | 180 × 24 × 51 | 0.15 s | 0.014 s | 0.047 s | 0.048 s | 10.6× | 6.1× |
| EthanolConcentration | 263 × 3 × 1751 | 11.6 s | 1.6 s | 5.5 s | 4.9 s | 7.2× | 3.7× |
| StandWalkJump | 15 × 4 × 2500 | 0.71 s | 0.14 s | 0.47 s | 0.47 s | 5.1× | 2.1× |

- The median ratio is **8.9×** (v1 was 4.6×), and every dataset is above the 4× target.
  The smallest margin is on the longest series (StandWalkJump, 5.1×).
- The v2 kernel is still bit-identical to the reference (`tests/test_numba.py`).
- Run-to-run noise is large on Hydra's side. EthanolConcentration measured 19.9 s,
  14.8 s and 11.6 s in three runs; the Hanner times are stable. The endpoint therefore
  needs the controlled timing harness of M0: an idle machine, fixed CPU frequency or
  thermal state if possible, interleaved repeats and several runs.
- With equally optimized kernels (both blocked v2), the Hanner recursion is about
  **3.5–4× faster than the dense ternary controls**. That is the speed-up attributable to
  the cotree structure itself; in v1 it was about 2×.
- Caveats: Hydra uses float32 torch tensors and the ternary arms use float64. Results
  come from one machine, one split and six datasets. The aeon Hydra time includes its
  numpy→torch conversion.
- Possible further steps: float32, int32 winner indices, fusing leaves into their parent
  node, and a C++ / SIMD kernel. None of these is needed for the 4× target on current
  evidence.

## Open design questions

Each of these must be fixed before pre-registration at M1. Record the decision here.

1. ~~**Hard-min counts are redundant for Hanner groups.**~~ **Resolved 2026-11-03: (a)
   and (c), see the decision log.** Hanner polytopes are centrally symmetric, so
   `argmin_v <x, v>` is the antipode of the argmax. Hydra's hard-min count of atom *j*
   therefore equals the hard-max count of `-j`. The independent random-ternary control is
   *not* symmetric, so its min counts carry extra information, which biases the comparison
   toward the control. Options considered:
   - (a) emit soft-max and hard-max counts only (**adopted**, `features="max"`, the default);
   - (b) replace the min counts with second-place counts (not adopted, because it breaks
     the O(l) single-winner recursion);
   - (c) add a *symmetrized* control (draw |V|/2 vertices and add their antipodes) as a
     second control (**adopted**, `kernel_family="control_sym"`).
2. **Feature-budget matching.** Groups are sampled per (dilation, input) until the atom
   count reaches `atom_budget` (default 256, which is Hydra's 8 kernels × 32 groups per
   input). This overshoots by less than one group. Decide whether the budget should match
   exactly (truncate the last group), and whether to match on atoms or on features.
3. **Patch centering.** *Resolved 2026-11-04:* `center="mixed"` is the default for every
   arm (odd groups centered; see "Mixed centering on all development datasets" and the
   decision log). It replaced the interim `center=False`. The original design
   motivated centering by shift invariance; the deviation goes into the pre-registration.
   A difference-only input remains a possible ablation.
4. **Cotree distribution.** Open knobs: `V_max ∈ {18, 24, 32, 64}`, `p_product`, and
   `contiguous` (merge only adjacent tap blocks, a time-local bias). Note that a product of
   two leaves (a square) and an ℓ1 sum of two leaves (a diamond) both have 4 vertices. Only
   products of larger subtrees use up the vertex budget.
5. **Multivariate channel selection.** The transform follows Hydra-MV: each group sums a
   random channel subset of size `clip(C // 2, 2, 8)`, drawn with replacement. Verify parity
   with aeon's implementation at M0.
6. **Ties on zero-padded patches.** All-zero patches, at the edges or on constant series,
   tie across every vertex and resolve to index 0. Hydra has the same artefact. Decide
   whether to exclude padded anchors.
7. ~~**Hardware for the timing claim.**~~ *Resolved 2026-11-04:* the Apple M4 (NEON) is
   the only official timing hardware; no x86 machine. Cache-miss rates then depend on
   Apple's CPU counters (pre-registration, decision 9.4).
8. ~~**Unequal-length UEA datasets.**~~ *Resolved 2026-11-04:* the endpoint uses the 26
   equal-length datasets, without padding (preregistration §3). Four of the 30 UEA datasets (CharacterTrajectories,
   InsectWingbeat, JapaneseVowels, SpokenArabicDigits) have series of unequal length.
   Neither aeon's `HydraClassifier` nor our transform accepts them, and the harness
   skips them for now. Options: pad or truncate to a common length (as in the
   equal-length UEA versions), or report on the 26 equal-length datasets
   (`--suite uea26`) and state the restriction in the pre-registration.
9. **Published resample splits.** Resamples 1–29 of the published Hydra results are not
   aeon's stratified resamples (see "Resampling harness"). Recreating the Java tsml
   resample files would allow paired per-resample comparisons with published results.
   This is not needed for the endpoint. The secondary comparison with other methods
   (preregistration §6) is therefore paired on per-dataset means.
10. **Pilot go/no-go rule.** *Resolved 2026-11-04:* with 5 datasets the Bayesian test is
    dominated by the prior, so the pilot decides on the mean difference instead: go if it is
    ≥ −0.5 pts against both Hydra and `control_sym`. The Bayesian test is reported but does
    not decide. The Bayesian H1 decision stays at M2.
11. ~~**High-channel datasets.**~~ On PEMS-SF (C = 963) Hanner and `control_sym` are about
    3 pts behind Hydra (pilot run 3 side check). With about 11 groups per block, each
    summing at most `clip(C // 2, 2, 8)` channels, the ternary arms see far fewer channels
    than Hydra's 32 groups. Options: more groups per block at the same atom budget
    (smaller cotrees), or a larger channel cap for high C. Decide before the M2 design
    freeze, on development datasets only, and apply it to all ternary arms alike.
    *Resolved 2026-11-04:* the default stays `max_channels=8`; raising the cap does not
    help on the mean, and Hydra keeps its lead on PEMS-SF at equal channel coverage. The
    high-channel gap is a known limitation for M2 (see "Diagnosis of high-channel
    datasets").

## Decision log

| Date | Decision | Rationale |
| :--- | :--- | :--- |
| 2026-11-03 | `atom_budget` default 256 per (dilation, input) | Matches aeon `HydraTransformer(8, 64)`, which splits its groups between the series and its difference |
| 2026-11-03 | Default features are soft-max + hard-max counts (`features="max"`). Hydra's soft-max + hard-min counts stay available as `features="hydra"` for an ablation. | For centrally symmetric groups the hard-min counts are a permutation of the hard-max counts, so they add no information. Hard-max counts fall out of the same O(l) recursion and need no second pass. |
| 2026-11-03 | Add a second, centrally symmetric control (`control_sym`). Antipodal pairs `(u, -u)` are drawn with the Hanner group's sparsity and take the same antipode permutation. Its argmin is the antipode of the argmax, so ties resolve as in Hanner groups. | Separates the cotree structure from central symmetry. Every comparison reports both controls. |
| 2026-11-03 | `control_sym` is the primary control of the endpoint. `control` is reported as a secondary comparison. | It is the stricter comparison: it matches the Hanner groups in size, sparsity *and* symmetry, so a win over it isolates the cotree structure. The independent control remains for continuity with the original design. Record the deviation in the M1 pre-registration. |
| 2026-11-03 | Resamples come from aeon `stratified_resample_data` with seed = resample id (resample 0 = default split), and every arm is seeded with the resample id | The tsml-eval convention. All arms share identical splits, so comparisons are paired per resample. Published results are compared only on per-dataset means (see "Resampling harness"). |
| 2026-11-03 | The Bayesian signed-rank test follows `baycomp`: prior strength 0.5, region probabilities as the fraction of posterior draws in which the region is most probable, `P(not worse) = P(rope) + P(right)` | Matches the authors' reference implementation, so results can be compared with the literature. Fix this definition in the pre-registration. |
| 2026-11-04 | `center="mixed"` is the default for all ternary arms (replaces `center=False`) | Over 34 development datasets: +0.11 pts against Hydra, and never more than 0.5 pts worse than either fixed setting on any of the three development sets. The decision rule was fixed before the run. Deviates from the original design; state it in the pre-registration. |
| 2026-11-04 | Fresh M1 pilot (run 3): MedicalImages, WordSynonyms, ScreenType, LSST, PhonemeSpectra × 30 resamples, rule unchanged. PEMS-SF is a side check that is reported but does not decide. | The run 1/2 pilot datasets were used twice. PhonemeSpectra (C = 11) replaces the proposed PEMS-SF (C = 963), decided before any result: with about 11 groups per block, each summing at most 8 channels, Hanner sees only a fraction of 963 channels (Hydra about 3× more), which would let one atypical dataset carry 20% of a 5-dataset decision. Running PEMS-SF alongside flags the high-channel bottleneck early; it is in M2 anyway. |
| 2026-11-04 | `center=False` is the default for all ternary arms (Hanner and both controls) | Diagnosis on 22 non-pilot datasets: centering costs 0.92 pts on multivariate data (+0.15 on univariate), and without it Hanner ties Hydra (+0.04) and beats both matched controls (+0.45 / +0.38). One rule for all data types, rather than a data-type rule chosen after seeing results. Deviates from the original design; state it in the pre-registration. |
| 2026-11-04 | M1 pilot go/no-go: mean difference ≥ −0.5 pts against Hydra and `control_sym` over the 5 pilot datasets × 30 resamples; the Bayesian test is reported only | With 5 datasets the posterior is dominated by the prior, so the H1 criterion is unreachable for a true difference near zero. The H1 decision itself stays at M2. |
| 2026-11-04 | `SparseScaler` uses the unbiased standard deviation (ddof=1) | Parity with aeon's `_SparseScaler`, where `torch.std` is unbiased. Results before 2026-11-04 (pilot run 1) used ddof=0; the effect is close to a uniform rescaling, which the ridge alpha search absorbs. |
| 2026-11-03 | The Hydra baseline is aeon 1.6.0 `HydraClassifier` (and `HydraTransformer` for timing), run by us under the same protocol | It reproduces aeon's published UCR results (mean diff +0.42 pts, gate ±0.5). It works for multivariate data. Using our own runs keeps the protocol and hardware identical across arms. |
| 2026-11-04 | M1 pilot run 3: **go** (+0.53 pts against Hydra, −0.10 against `control_sym`). The study continues to the design freeze and M2. | The pre-declared rule is met on 5 fresh datasets. Recorded caveat: without ScreenType the difference to Hydra would be −0.87, so M2 is genuinely open. PEMS-SF (side check) is −3.08 behind Hydra, for `control_sym` too. |
| 2026-11-04 | `max_channels` stays 8 (as in Hydra) for all ternary arms; `"auto"` and fixed caps remain available for ablations | The pre-declared development rule failed (+0.01 pts, needed +0.5) over the four used datasets with C ≥ 18, so no confirmation run. Hydra with Hanner's group count keeps its PEMS-SF lead, so channel coverage is not the cause. Either way only the 7 UEA datasets with C ≥ 18 would be affected (the UCR part cannot be; first recorded wrongly as "the primary (univariate) endpoint is unaffected"). |
| 2026-11-04 | The Apple M4 is the only official timing hardware for the endpoint | All development timing was done there; the 4× criterion is a ratio measured on one machine. Consequence: cache-miss rates need Apple's CPU counters, as `perf` is unavailable. |
| 2026-11-04 | **Pre-registration frozen** ([preregistration.md](preregistration.md)): 112 UCR + 26 equal-length UEA datasets, one pooled Bayesian signed-rank test, 3 timing passes on the Apple M4; other methods from aeon's published reference results (UCR only), SAX-style word counts dropped, cache misses only if Apple's counters are reliable | Confirmed choices of preregistration §9. Frozen before any M2 run; the timing and benchmark harnesses (predictions, balanced accuracy, macro-F1) and the `--subset` analyses are part of the frozen code. |
| 2026-11-06 | M2 accuracy: Hanner-Hydra is **not worse** than Hydra (Δ −0.06, P(not worse) 0.988) and `control_sym` (Δ +0.04, P 1.000) over the 138 pre-registered datasets; decision conditions 1 and 2 hold | Pre-registered analysis, no failures or exclusions. Recorded caveats: equivalent rather than better (47/86 W/L against Hydra); UEA alone is not shown (Δ −0.54, P 0.22), a gap that `control_sym` shares (−0.77). H1 awaits the timing endpoint. |
| 2026-11-06 | Deviation D1 (prereg §11): `analyze.py` sorts the resamples before averaging | The resample order came from set iteration, which varies between processes, so dataset means differed in the last bit and the exact-tie W/T/L counts changed by one or two between runs. The means, probabilities and decisions are unchanged. |
| 2026-11-07 | Timing pass 0 repeated, single-process, as prereg §4 requires; the first pass 0 is kept as a sensitivity check | The first pass 0 overlapped active use of the Mac on 2026-11-06. The rerun agrees with it (median time ratio 1.001 Hydra, 1.008 Hanner), and the endpoint is 11.23× with either. |
| 2026-11-07 | **H1 accepted**: not worse than Hydra (P 0.988) and `control_sym` (P 1.000) in accuracy, and a median transform-time ratio of 11.23× (≥ 4) on the Apple M4 | All three pre-registered conditions hold (prereg §5). Report with the caveats: equivalent rather than better, UEA alone not shown, and the cotree's contribution is speed (3.8× over the matched controls), not accuracy. |
| 2026-11-07 | Cache-miss rates are dropped (prereg §6 fallback) | Developer mode is off on this managed Mac. `xctrace` CPU Counters recorded data but produced no readable trace, and the `kpc` interface needs root. |
| 2026-11-07 | Secondary analyses: α = 0.05 for the Wilcoxon–Holm tests and CD diagrams (not fixed in the prereg); CD cliques drawn with aeon's convention; "practically worse" label added as the mirror of *better*; matplotlib added to the `bench` extra for the diagrams | Chosen before running the tests. aeon's cliques use one-sided tests at α/(k−1), so the table, not the diagram, is the pre-registered test. matplotlib only adds packages to the lock, and every pinned version is unchanged. |
| 2026-11-08 | aeon estimators are thin wrappers that delegate to the reference `HannerHydraTransform`, in a separate module (`hanner_hydra.estimators`) behind an `aeon` extra; names follow aeon's `HydraTransformer` / `HydraClassifier` | The frozen transform stays untouched, and the wrappers reproduce it bit for bit (tests, and the stored M2 predictions on 6 datasets × 2 arms). The core package keeps needing only NumPy and scikit-learn. Both pass aeon 1.6.0's `parametrize_with_checks`. The Numba layout is built in `fit`, because aeon requires `transform` not to change state. |
| 2026-11-08 | aeon estimators, review fixes: CI runs a core job (no aeon) and an aeon job; the estimators are in `__all__` only when aeon is installed; the `aeon` extra is `>=1.6,<2` and `bench` depends on it; the M2-identity check is a test (`HH_CHECK_M2=1`) | Without aeon in CI, `tests/test_estimators.py` was skipped there, and `from hanner_hydra import *` failed on the core install. The wrappers implement aeon's private hooks, so a new major version needs a re-check. The wrapper relies on the reference's private `_packed` cache, now tested directly. No change to the reference transform or any result. |
| 2026-11-08 | Results bundle: the results go out as a deterministic tarball attached to the GitHub release `results-v1`; it holds the M2, ablation, timing, memory and M0-gate results with all per-case predictions (29,436 files, 7.9 MB), under their `results/` paths, with a SHA-256 manifest; third-party results (aeon's published results, the Hydra paper's) are left out and only their checksums recorded | Extracted in a checkout of the release commit, `analyze.py`, `secondary.py all` and the timing and memory summaries reproduce every stored CSV and PNG byte for byte; the CD-diagram PDFs differ only in their embedded creation date (checked in a clean worktree; the re-downloaded published results matched the recorded checksums). Predictions let readers compute other metrics without the ~30 h of compute. Not redistributing third-party results avoids licence questions; the checksums detect upstream changes. |
| 2026-11-08 | Results bundle, review fixes: `release.py` refuses to build with uncommitted changes or missing third-party files (override flags exist, not for a release); `--verify` reports mismatched, missing and unlisted files instead of stopping on the first missing one; `--check-third-party` compares new downloads with `RELEASE.json`; tests in `tests/test_release.py` | A dirty or incomplete build could have been released by mistake, and a damaged archive gave a traceback. The bundle content is unchanged: the rebuilt manifest and third-party checksums are identical to the first build. |
| 2026-11-08 | Results licence: CC BY 4.0 (the code stays BSD-3-Clause); stated in the bundle README with a citation, in `RELEASE.json` (`"license": "CC-BY-4.0"`) and in the repository README | CC BY 4.0 is the usual licence for research data and Zenodo's default, so it fits the later DOI; it requires credit, which CC0 would only request. Set before the first release, so no copy was ever published under BSD. The rights holder is Jörg Bahlmann, as in `LICENSE`. The release is named `results-v1` and cited as *Hanner-Hydra benchmark results*. |
| 2026-11-08 | Results bundle `results-v1` built from a clean checkout (`RELEASE.json`: `dirty: false`) and checked with `--verify` | The checksum of the uploaded archive is in the GitHub release notes; the manifest (`MANIFEST.sha256`) covers every result file. |
| 2026-11-08 | Paper: arXiv preprint (cs.LG, cross-list stat.ML), then Data Mining and Knowledge Discovery; source in `paper/`, plain `article` class with natbib for now, Springer `sn-jnl` for the journal | DMKD published Hydra, ROCKET, MultiRocket, QUANT and the bake-offs, so its reviewers know the protocol. arXiv needs TeX source and compiles `article` as is. The literature review (`paper/literature-review.md`) found no prior structured or polytope-based kernel groups; the closest work, NanoHydra (binary kernels, IJCNN 2025), evaluates every kernel. The Mahler result is cited as an unrefereed claim; the method uses only the combinatorial structure. |
| 2026-11-08 | Paper, review fixes: paper tables and figures generated by `experiments/paper_tables.py` from `results/` (committed under `paper/tables`, `paper/figures`); title keeps "multiplication-free", with the Method section stating that it covers the group evaluation (centering by the scaled form $\ell x_i - s$; scaler and ridge multiply); outline macro renamed `\paperoutline`; bibliography checked against Crossref, arXiv and JMLR (4 entries still flagged) | No number in the paper is typed by hand: the script reads the unrounded results, since the rounded `dataset_means.csv` turns near-ties into ties (it gave W/T/L 47/6/85 instead of 47/5/86). NanoHydra's Table I averages "over 100" UCR datasets without a list, uses best of 20 runs, and copies Hydra's 85.31 % from the WEASEL 2.0 paper; the paper says so instead of claiming the same datasets. |
| 2026-11-08 | Two corrections to recorded numbers, found while writing the results section: the largest per-dataset time ratio is 23.15× (StarLightCurves), not 23.71×; the smallest memory ratio is 1.17× (DuckDuckGeese), not "at least 1.3×" | The 23.71× came from `summary_with_nonidle_pass0.csv`, which includes the discarded non-idle pass; the endpoint summary gives 23.15× (the median, 11.23×, is the same in both). The 1.3× was wrong: DuckDuckGeese uses 163.7 MB (Hydra) vs 140.5 MB (Hanner). No decision or other number changes. |
