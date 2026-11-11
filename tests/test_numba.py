"""The Numba backend must reproduce the NumPy reference bit for bit."""

import numpy as np
import pytest

pytest.importorskip("numba")

from hanner_hydra import HannerHydraTransform


@pytest.mark.parametrize("family", ["hanner", "control", "control_sym"])
@pytest.mark.parametrize("features", ["max", "hydra"])
@pytest.mark.parametrize("center", [True, False, "raw", "diff", "mixed"])
@pytest.mark.parametrize("increments", [False, True])
@pytest.mark.parametrize("shape", [(5, 70), (4, 3, 41), (3, 12, 9), (3, 1, 200)])
def test_numba_matches_reference_exactly(
    family: str, features: str, center: bool | str, increments: bool, shape: tuple[int, ...]
) -> None:
    X = np.random.default_rng(7).standard_normal(shape)
    kw = {
        "kernel_family": family,
        "features": features,
        "center": center,
        "increments": increments,
        "atom_budget": 48,
        "random_state": 1,
    }
    t = HannerHydraTransform(backend="numpy", **kw).fit(X)
    ref = t.transform(X)
    fast = HannerHydraTransform(backend="numba", **kw).fit(X).transform(X)
    np.testing.assert_array_equal(fast, ref)


def test_numba_exact_on_tie_heavy_input() -> None:
    # Integer-valued and constant series create many exact ties.
    rng = np.random.default_rng(0)
    X = rng.integers(-2, 3, (4, 2, 60)).astype(np.float64)
    X[0] = 1.0  # constant case
    for family in ("hanner", "control", "control_sym"):
        kw = {"kernel_family": family, "features": "hydra", "atom_budget": 64, "random_state": 3}
        ref = HannerHydraTransform(backend="numpy", **kw).fit(X).transform(X)
        fast = HannerHydraTransform(backend="numba", **kw).fit(X).transform(X)
        np.testing.assert_array_equal(fast, ref)


def test_refit_rebuilds_packed_layout() -> None:
    rng = np.random.default_rng(1)
    t = HannerHydraTransform(backend="numba", atom_budget=32, random_state=0)
    t.fit(rng.standard_normal((3, 50)))
    t.transform(rng.standard_normal((2, 50)))
    X2 = rng.standard_normal((3, 2, 120))  # different shape -> different layout
    ref = HannerHydraTransform(backend="numpy", atom_budget=32, random_state=0).fit(X2)
    np.testing.assert_array_equal(t.fit(X2).transform(X2), ref.transform(X2))


@pytest.mark.parametrize("max_channels", [16, "auto"])
def test_numba_exact_with_max_channels(max_channels) -> None:
    X = np.random.default_rng(3).standard_normal((4, 70, 60))
    for family in ("hanner", "control_sym", "control"):
        kw = {"kernel_family": family, "max_channels": max_channels, "random_state": 2}
        ref = HannerHydraTransform(backend="numpy", **kw).fit(X).transform(X)
        fast = HannerHydraTransform(backend="numba", **kw).fit(X).transform(X)
        np.testing.assert_array_equal(fast, ref)


def test_numba_exact_with_diff_share() -> None:
    X = np.random.default_rng(2).standard_normal((4, 3, 90))
    for family in ("hanner", "control_sym"):
        kw = {"kernel_family": family, "diff_share": 0.75, "atom_budget": 64, "random_state": 5}
        ref = HannerHydraTransform(backend="numpy", **kw).fit(X).transform(X)
        fast = HannerHydraTransform(backend="numba", **kw).fit(X).transform(X)
        np.testing.assert_array_equal(fast, ref)
