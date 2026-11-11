# Hanner-Hydra: pre-registration of the primary endpoint

**Status: FROZEN on 2026-11-04.** The code, the analysis and this document were frozen
at commit `3b0cd46` of the private development repository, as an annotated git tag. No
M2 result existed at the time of freezing. That history is not public, so the freeze
date is not independently timestamped.

**Public edition.** This text differs from the frozen document only in wording:
references to internal planning documents and project labels were removed or reworded
(the original design is now described where it was cited, Sections 5 and 8), the phase names
M0–M2 are explained below, the development log replaces the internal work plan as the
linked record, and Section 11 was added after the freeze. No design choice, endpoint,
threshold or analysis was changed. The SHA-256 of the frozen document
is `33213ced23ed9558c6a296483e0e9c3960f2e2aadf6ea9d2a4b60572bdeb9363`, also recorded in
`RELEASE.json` of the results bundle.

This document fixes the design, the primary endpoint, the analysis and the decision rule
before the full evaluation. The study ran in phases: M0 infrastructure, M1 pilots and
this pre-registration, M2 the full evaluation. Everything not fixed here is secondary
and exploratory. Deviations from the original study design are listed in Section 8, the
choices confirmed at the freeze in Section 9. Background and the history of every
choice are in [development-log.md](development-log.md).

## 1. Hypotheses

- **H1:** Hanner-Hydra is *not worse* than both Hydra and the symmetric theory-matched
  control `control_sym`, and its single-core CPU transform is at least 4× faster than
  Hydra's (median ratio across datasets).
- **H0:** Hanner-Hydra is practically worse than Hydra or than `control_sym`, or it is
  less than 4× faster than Hydra.

## 2. Frozen design

All arms come from the code at the freeze; dependency versions are pinned by
`uv.lock` (aeon 1.6.0, NumPy 2.3.5, scikit-learn 1.9.1, Numba 0.63.1, torch 2.15.0).

**Hanner-Hydra** (`make_classifier(kernel_family="hanner")`):

| Parameter | Value | Meaning |
| :--- | :--- | :--- |
| `kernel_length` | 9 | taps per kernel |
| `max_vertices` | 32 | largest group (Hanner polytope vertex count) |
| `p_product` | 0.5 | probability of an ℓ∞ product at a cotree merge |
| `contiguous` | False | merges are not restricted to adjacent taps |
| `atom_budget` | 256 | atoms per (dilation, input), as Hydra's 8 × 32 |
| `use_diff`, `diff_share` | True, 0.5 | series and first difference, equal budgets |
| `center` | `"mixed"` | odd-numbered groups in each block see mean-centred patches |
| `increments` | False | groups act on the patches themselves |
| `max_channels` | 8 | channel subset size `clip(C // 2, 2, 8)`, as Hydra |
| `features` | `"max"` | soft-max and hard-max counts per atom |

Dilations, zero padding and the per-(dilation, input) dictionaries follow Hydra. The
features go through Hydra's sparse scaler (unbiased standard deviation, as aeon) and
`RidgeClassifierCV(alphas=logspace(-3, 3, 10))`.

**Controls.** `control_sym` (primary) and `control` (secondary) use the identical
pipeline and settings. For every Hanner group they draw a random ternary group with the
same number of vertices and non-zeros per vertex; `control_sym` draws it in antipodal
pairs, so it is centrally symmetric like the Hanner group.

**Hydra.** aeon 1.6.0 `HydraClassifier(n_kernels=8, n_groups=64)` with its defaults.
It reproduces aeon's published UCR results (mean difference +0.42 pts; development-log.md,
"Hydra baseline reproduction").

**Seeds.** Every arm is seeded with the resample id.

## 3. Datasets and resamples

- **Datasets:** the 112 UCR datasets of the Hydra evaluation (aeon
  `univariate_equal_length`) and the 26 equal-length UEA datasets (aeon
  `multivariate_equal_length`): **138 datasets**. The four unequal-length UEA datasets
  are not used and not padded.
- **Resamples:** ids 0–29. Resample 0 is the default train/test split; resample *i* pools
  train and test and redraws a stratified split with the original class counts, using
  aeon `stratified_resample_data(random_state=i)`. All arms see identical splits.
- **Runs:** both suites into one results file,
  `experiments/benchmark.py --suite ucr112 --resamples 30 --tag m2` and the same with
  `--suite uea26`, default arms `hanner control_sym control hydra`. Accuracy runs may use
  parallel single-threaded workers; their times are not the timing endpoint. Every row
  stores accuracy, balanced accuracy and macro-F1, and the per-case test predictions are
  kept for the results bundle.

## 4. Primary endpoint and analysis

**Accuracy.**

- Score of an arm on a dataset: mean top-1 test accuracy over the 30 resamples, in
  percentage points. Δ = Hanner-Hydra minus comparator.
- Test: one Bayesian signed-rank test (Benavoli et al., 2017) over all 138 datasets,
  ROPE [−0.5, +0.5], as implemented in `hanner_hydra.stats` (baycomp convention: prior
  strength 0.5 on a pseudo-observation at 0, region probabilities as the share of 50,000
  posterior draws in which that region is the most probable, seed 0).
- *Not worse* against a comparator: `P(rope) + P(right) ≥ 0.95`.
- Run: `experiments/analyze.py --tag m2`.

**Transform time** (single core, Apple M4):

- Hardware: the Apple M4 development machine (4 performance and 6 efficiency cores,
  32 GB, macOS 27.2), on mains power, Low Power Mode off, no other user workload. It is
  the only official timing hardware. The harness records the power source and Low Power
  Mode and warns on a violation; a pass run in violation is repeated.
- Threads: one thread per process (`OMP` / `MKL` / `OpenBLAS` / `Numba` pinned to 1,
  `torch.set_num_threads(1)`). macOS has no core affinity; the dispersion between
  passes is reported.
- Timed region: `transform(X_test)` only, on the default split (resample 0), after
  `fit(X_train)`. The input is already a NumPy float64 array in memory. Hydra is aeon's
  `HydraTransformer(n_kernels=8, n_groups=64)` through its public API, including its
  conversion to float32 torch tensors and back. Hanner-Hydra uses the Numba kernel
  (`backend="numba"`, float64), bit-identical to the NumPy reference. Neither the
  scaler nor the ridge classifier is timed.
- Procedure per dataset: one untimed warm-up `transform(X_test)` per arm (JIT
  compilation, caches), then 7 timed calls per arm, interleaved across arms in a
  rotating order, with `gc.collect()` before each call. Per arm, the median of the 7.
- Three passes in separate processes, with a seeded random dataset order per pass
  (seed 0). The time of an arm on a dataset is the median of its three pass medians.
- Endpoint: per dataset the ratio Hydra time / Hanner-Hydra time; criterion: the median
  ratio over the 138 datasets is ≥ 4.
- Run: `experiments/timing.py --suite ucr112 --suite uea26 --tag endpoint`, then
  `--tag endpoint --summary`.

## 5. Decision rule

H1 is accepted if and only if all three hold:

1. Not worse than Hydra: `P(Δ ≥ −0.5) ≥ 0.95`.
2. Not worse than `control_sym`: `P(Δ ≥ −0.5) ≥ 0.95`.
3. Median transform-time ratio Hydra / Hanner-Hydra ≥ 4.

Otherwise H0 is retained and the report states which condition failed. Each comparison
is also labelled as in the original design: *better* if `P(Δ > 0.5) ≥ 0.95`, *not worse*
if `P(Δ ≥ −0.5) ≥ 0.95`, *practically equivalent* if `P(|Δ| ≤ 0.5) ≥ 0.95`, otherwise
inconclusive.

**Completeness.** A dataset enters the accuracy analysis only if every arm completed all
30 resamples, and the timing analysis only if every arm completed all three passes.
Failures (errors, memory) are reported with their cause. Settings are not changed for
individual datasets, and failed datasets are not replaced.

**After the freeze** the design and the analysis are not changed. Any later change is a
deviation: it is documented, and the pre-registered analysis is still reported.

## 6. Secondary analyses (reported, not deciding)

- **Subsets** (`analyze.py --subset`): UCR only (`ucr112`), UEA only (`uea26`), and the
  93 datasets that informed no design choice (`fresh`: 84 UCR, 9 UEA; Section 10).
- Hanner-Hydra against `control` (the control of the original design).
- `control_sym` against Hydra, and the transform-time ratio of each control to
  Hanner-Hydra (the speed attributable to the cotree recursion).
- Two-sided Wilcoxon signed-rank tests with Holm correction and critical-difference
  diagrams.
- Balanced accuracy and macro-F1, analysed as accuracy.
- **Other methods:** WEASEL 2.0, MiniROCKET, MultiROCKET and HIVE-COTE 2.0 from aeon's
  published 30-resample reference results (`aeon.benchmarking.results_loaders`),
  compared with Hanner-Hydra on per-dataset means. aeon publishes these results for the
  112 UCR datasets only, so this comparison covers UCR only. The published resample
  splits differ from ours (development-log.md, open question 9), so the comparison is paired by
  dataset, not by resample.
- Peak transform memory; fit times from the timing passes; the distribution of the
  per-dataset time ratios with a bootstrap 95% interval of the median.
- L1/L2 cache-miss rates, if Apple's CPU counters on the M4 can be read reliably;
  otherwise this metric is dropped and the report says so.
- Ablations, explicitly exploratory: `center` ∈ {False, True}, `max_channels` ∈
  {32, `"auto"`}, `features="hydra"`.

## 7. Harness at the freeze

Both harnesses that implement this protocol are part of the frozen code:

- `experiments/timing.py`: the procedure of Section 4 (interleaved repeats, three passes
  in separate processes, seeded dataset orders, per-pass CSVs, resumable, `--summary`).
- `experiments/benchmark.py`: accuracy, balanced accuracy and macro-F1 per (dataset,
  resample, arm), plus per-case predictions under `predictions/`.
- `experiments/analyze.py`: the primary test and the subsets of Section 6.

## 8. Deviations from the original study design

| Original design | Pre-registered | Reason |
| :--- | :--- | :--- |
| Control: random ternary groups, same sizes and sparsity | Primary control `control_sym`: the same, but centrally symmetric; the original control is secondary | Separates the cotree structure from central symmetry, the stricter comparison (decision log 2026-11-03) |
| Centre the input patches | `center="mixed"`: half the groups centred | Centring helps some data and hurts other data; mixed was the most robust setting over 34 development datasets (decision log 2026-11-04) |
| Hydra's hard-min counts | Hard-max counts | For centrally symmetric groups the hard-min counts are a permutation of the hard-max counts (decision log 2026-11-03) |
| All 30 UEA datasets | 26 equal-length UEA datasets | Neither aeon's Hydra nor this transform accepts unequal lengths; the 26 are the set of the multivariate bake-off (Ruiz et al., 2021) |
| M1: continue only if the pilot beats both comparators | Mean difference ≥ −0.5 pts against Hydra and `control_sym` on 5 datasets | With 5 datasets the Bayesian test is prior-dominated; the bar matches the not-worse endpoint (decision log 2026-11-04) |
| C++ / Numba SIMD kernel | Numba kernel, no fastmath, bit-identical to the reference | Median ratio 12.9× on the pilot already; C++ was planned only if Numba fell short |
| Timing on AVX-512 and/or NEON | Apple M4 (NEON) only | Decision 2026-11-04 |
| Accuracy against WEASEL 2.0, MiniROCKET, MultiROCKET, HIVE-COTE 2.0 | From aeon's published reference results, UCR only, paired on dataset means | New 30-resample runs would take weeks (HIVE-COTE 2.0); aeon publishes no UEA reference results for them |
| SAX-style word counts | Dropped | No standard estimator to run or cite |
| L1/L2 cache miss rates (secondary) | Only if Apple's CPU counters can be read reliably | No `perf` on macOS |
| Negative log-likelihood | Not reported | Ridge classifiers (Hydra and all arms here) give no probabilities |

## 9. Choices confirmed at the freeze (2026-11-04)

1. The 26 equal-length UEA datasets, no padding of the four unequal-length ones.
2. One pooled primary test over all 138 datasets; UCR, UEA and the 93 fresh datasets
   as secondary analyses.
3. Other methods from aeon's published reference results, paired on per-dataset means;
   SAX-style word counts dropped.
4. Cache misses through Apple's CPU counters, dropped if they cannot be read reliably.
5. The Apple M4 as the only official timing hardware.

## 10. Datasets that informed the design

Used for accuracy-based design decisions in M0–M1 (development, diagnosis and pilot
runs; Hydra-only reproduction runs and smoke tests are not counted). The same list is
`DESIGN_DATASETS` in `experiments/analyze.py`.

- **UCR (28):** Adiac, Beef, Car, CinCECGTorso, Computers, DistalPhalanxOutlineCorrect,
  ECG5000, EOGVerticalSignal, Earthquakes, FiftyWords, Ham, Haptics, Herring,
  InlineSkate, InsectWingbeatSound, Lightning2, Lightning7, Meat, MedicalImages, OliveOil,
  Rock, ScreenType, SemgHandGenderCh2, Strawberry, SwedishLeaf, Wine, WordSynonyms, Worms.
- **UEA (17):** ArticularyWordRecognition, Cricket, ERing, EthanolConcentration,
  FingerMovements, HandMovementDirection, Handwriting, Heartbeat, LSST, Libras, NATOPS,
  PEMS-SF, PhonemeSpectra, RacketSports, SelfRegulationSCP1, SelfRegulationSCP2,
  UWaveGestureLibrary.

## 11. Deviations after the freeze

Changes made after the freeze (Section 5). The analysis as frozen is still reported
where its results differ.

| ID | Date | Change | Reason and effect |
| :--- | :--- | :--- | :--- |
| D1 | 2026-11-06 | `dataset_means` in `experiments/analyze.py` iterates the (dataset, resample) pairs in sorted order | The frozen code took the resample order from set iteration, which depends on Python's per-process hash seed. Dataset means then differed in the last floating-point bit between runs, and the W/T/L counts, which count exact ties, changed by one or two datasets. The means, the Bayesian probabilities (to 3 decimals) and every decision are identical before and after the fix. Found on the M2 run, after its completion. |
