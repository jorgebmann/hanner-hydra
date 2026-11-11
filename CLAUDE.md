# Hanner-Hydra

Research code for Hanner-Hydra. The design history, results and open design questions are
in `docs/development-log.md`. Record design decisions in its decision log.

## Commands
- `uv sync` installs the core package; add `--extra bench --extra fast` for aeon/torch/numba
  (`--extra aeon` alone is enough for the aeon estimators in `hanner_hydra.estimators`)
- `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`
  (`HH_CHECK_M2=1 uv run pytest -k m2` also checks the aeon classifier against the stored M2
  predictions; needs the local data and `results/benchmark/m2`)
- `uv run python experiments/pilot.py <UCR/UEA names...>` (needs the bench extra)
- `uv run python experiments/timing.py [datasets...] [--suite ucr112 --suite uea26] --tag T`
  (3 passes, resumable; `--summary` prints the endpoint; needs the bench and fast extras)
- `uv run python experiments/reproduce_hydra.py [--summary]` (Hydra baseline gate; resumable, about 50 min)
- `uv run python experiments/benchmark.py [--suite ucr112|uea26] --resamples 30 --jobs 8 --tag T`
  (resumable) and `uv run python experiments/analyze.py --tag T [--published] [--subset ...]`
- `uv run python experiments/secondary.py all --tag m2 --timing-tag endpoint` (prereg §6 analyses)
  and `uv run python experiments/memory.py --suite ucr112 --suite uea26 --jobs 3 --tag T`
- `uv run --extra aeon python experiments/paper_tables.py --tag m2` writes `paper/tables/` and
  `paper/figures/`; build the paper with `cd paper && tectonic main.tex`
- `uv run python experiments/release.py [--verify dist/<archive> | --check-third-party]` builds
  the results bundle (deterministic; refuses uncommitted changes, so build from a clean
  checkout of the release commit)

## Conventions
- The endpoint is frozen in `docs/preregistration.md` (2026-11-04). Do not change the
  frozen defaults, harnesses or analysis; a later change is a documented deviation.
- `src/hanner_hydra/estimators.py` (aeon) only wraps the reference; it must stay identical to
  `make_classifier` / `HannerHydraTransform` (`tests/test_estimators.py`).
- `src/hanner_hydra/transform.py` is the NumPy *reference* implementation. Fast kernels
  (`_numba.py`, later C++) must reproduce it bit for bit (`tests/test_numba.py`). Timing
  claims are only made on the fast kernel.
- Canonical vertex order and tie-breaking are documented in `cotree.py`. The support
  recursion must match `np.argmax` over `vertices()`.
- Every comparison includes both theory-matched controls (`kernel_family="control"` and
  `"control_sym"`). The default features are soft-max + hard-max (`features="max"`).
- `openai/math` citations are pinned to commit `4eaee87`.
- `data/` and `results/` are git-ignored.
