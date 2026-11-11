"""aeon-compatible estimators: aeon's estimator checks, and identity with the reference."""

from __future__ import annotations

import csv
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("aeon")

from aeon.testing.estimator_checking import parametrize_with_checks

from hanner_hydra import HannerHydraTransform, make_classifier
from hanner_hydra.estimators import HannerHydraClassifier, HannerHydraTransformer
from hanner_hydra.transform import _resolve_backend

BACKENDS = ["numpy"] + (["numba"] if _resolve_backend("auto") == "numba" else [])


@parametrize_with_checks([HannerHydraTransformer, HannerHydraClassifier])
def test_aeon_checks(check):
    check()


@pytest.mark.parametrize("cls", [HannerHydraTransformer, HannerHydraClassifier])
def test_same_parameters_and_defaults_as_reference(cls):
    ref = inspect.signature(HannerHydraTransform).parameters
    ours = inspect.signature(cls).parameters
    for name, p in ref.items():
        assert name in ours, name
        assert ours[name].default == p.default, name


def _data(n_channels: int, n_cases: int = 24, length: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.arange(n_cases) % 3
    X = rng.standard_normal((n_cases, n_channels, length))
    X += y[:, None, None] * np.sin(np.linspace(0, 6, length))  # learnable signal
    return X, y


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kernel_family", ["hanner", "control", "control_sym"])
@pytest.mark.parametrize("n_channels", [1, 3, 20])
def test_transformer_identical_to_reference(backend, kernel_family, n_channels):
    X, _ = _data(n_channels)
    params = {"kernel_family": kernel_family, "backend": backend, "random_state": 7}
    ref = HannerHydraTransform(**params).fit(X[:16]).transform(X[16:])
    ours = HannerHydraTransformer(**params).fit(X[:16]).transform(X[16:])
    assert ours.dtype == np.float64
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.skipif("numba" not in BACKENDS, reason="needs numba")
def test_numba_layout_built_in_fit_and_transform_keeps_state():
    """aeon requires ``transform`` not to change state; the wrapper relies on the reference's
    private layout cache (``_packed``), so a rename there must fail here, not silently."""
    X, _ = _data(3)
    est = HannerHydraTransformer(backend="numba", random_state=0).fit(X[:16])
    assert "_packed" in est.transform_.__dict__
    before = dict(est.transform_.__dict__)
    est.transform(X[16:])
    assert est.transform_.__dict__.keys() == before.keys()
    assert all(est.transform_.__dict__[k] is v for k, v in before.items())


def test_package_exports_estimators_with_aeon():
    import hanner_hydra

    assert hanner_hydra.HannerHydraClassifier is HannerHydraClassifier
    assert hanner_hydra.HannerHydraTransformer is HannerHydraTransformer
    assert {"HannerHydraClassifier", "HannerHydraTransformer"} <= set(hanner_hydra.__all__)
    assert {"HannerHydraClassifier", "HannerHydraTransformer"} <= set(dir(hanner_hydra))


def test_transformer_accepts_2d_univariate():
    X, _ = _data(1)
    a = HannerHydraTransformer(random_state=1).fit_transform(X)
    b = HannerHydraTransformer(random_state=1).fit_transform(X[:, 0, :])
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("n_channels", [1, 4])
def test_classifier_identical_to_make_classifier(n_channels):
    X, y = _data(n_channels, n_cases=36)
    labels = np.array(["a", "b", "c"])[y]  # string labels go through aeon's encoding
    ref = make_classifier(random_state=3).fit(X[:24], labels[:24]).predict(X[24:])
    clf = HannerHydraClassifier(random_state=3).fit(X[:24], labels[:24])
    np.testing.assert_array_equal(clf.predict(X[24:]), ref)
    assert set(clf.classes_) == {"a", "b", "c"}
    proba = clf.predict_proba(X[24:])
    assert proba.shape == (12, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_classifier_class_weight_matches_ridge_option():
    X, y = _data(2, n_cases=36)
    y = np.where(np.arange(36) < 30, y, 2)  # imbalanced: class 2 dominates
    ref = make_classifier(random_state=5)
    ref.set_params(ridgeclassifiercv__class_weight="balanced")
    ref_pred = ref.fit(X[:24], y[:24]).predict(X[24:])
    clf = HannerHydraClassifier(random_state=5, class_weight="balanced").fit(X[:24], y[:24])
    np.testing.assert_array_equal(clf.predict(X[24:]), ref_pred)
    assert clf.pipeline_[-1].class_weight == "balanced"


# The stored M2 predictions (resample 0, default split): the wrapper must reproduce them.
# Needs the local UCR/UEA data and results/benchmark/m2 (both git-ignored), so it is skipped
# in CI. Opt in with HH_CHECK_M2=1 (about 20 s).
ROOT = Path(__file__).resolve().parents[1]
M2_PREDICTIONS = ROOT / "results" / "benchmark" / "m2" / "predictions"
M2_DATASETS = ["GunPoint", "ArrowHead", "Coffee", "BasicMotions", "NATOPS", "PEMS-SF"]


@pytest.mark.skipif(os.environ.get("HH_CHECK_M2") != "1", reason="opt in with HH_CHECK_M2=1")
@pytest.mark.parametrize("arm", ["hanner", "control_sym"])
@pytest.mark.parametrize("name", M2_DATASETS)
def test_classifier_reproduces_m2_predictions(name, arm):
    stored = M2_PREDICTIONS / name / "r0" / f"{arm}.csv"
    if not stored.exists() or not (ROOT / "data" / name).exists():
        pytest.skip(f"no local data or M2 predictions for {name}")
    from aeon.datasets import load_classification

    config = json.loads((M2_PREDICTIONS.parent / "config.json").read_text())
    X_train, y_train = load_classification(name, split="train", extract_path=ROOT / "data")
    X_test, y_test = load_classification(name, split="test", extract_path=ROOT / "data")
    clf = HannerHydraClassifier(
        kernel_family=arm,
        features=config["features"],
        atom_budget=config["atom_budget"],
        random_state=0,  # every arm is seeded with the resample id
    ).fit(X_train, y_train)
    with stored.open() as f:
        rows = list(csv.DictReader(f))
    np.testing.assert_array_equal(np.asarray(y_test).astype(str), [r["y_true"] for r in rows])
    np.testing.assert_array_equal(clf.predict(X_test).astype(str), [r["y_pred"] for r in rows])
