"""aeon-compatible Hanner-Hydra estimators (needs the ``aeon`` extra).

* :class:`HannerHydraTransformer`: an aeon ``BaseCollectionTransformer`` around the
  reference :class:`~hanner_hydra.HannerHydraTransform`.
* :class:`HannerHydraClassifier`: an aeon ``BaseClassifier``, transform ->
  ``SparseScaler`` -> ``RidgeClassifierCV`` exactly as :func:`~hanner_hydra.make_classifier`
  (and aeon's ``HydraClassifier``).

Both take the parameters of ``HannerHydraTransform`` with the same defaults (the frozen
configuration, ``docs/preregistration.md`` §2) and delegate to it, so their output is
identical to the reference: the same features bit for bit, the same predictions. They add
aeon's input handling (2D / 3D NumPy arrays, lists of equal-length 2D arrays, pandas
DataFrames), validation, tags and the aeon estimator test suite.

Example::

    from aeon.datasets import load_classification
    from hanner_hydra.estimators import HannerHydraClassifier

    X_train, y_train = load_classification("GunPoint", split="train")
    X_test, y_test = load_classification("GunPoint", split="test")
    clf = HannerHydraClassifier(random_state=0).fit(X_train, y_train)
    accuracy = (clf.predict(X_test) == y_test).mean()
"""

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
from aeon.classification.base import BaseClassifier
from aeon.transformations.collection.base import BaseCollectionTransformer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline

from hanner_hydra.transform import (
    Backend,
    Centering,
    FeatureSet,
    HannerHydraTransform,
    KernelFamily,
    SparseScaler,
    _resolve_backend,
)

__all__ = ["HannerHydraClassifier", "HannerHydraTransformer"]

_PARAMS_DOC = """
    kernel_family : {"hanner", "control", "control_sym"}, default="hanner"
        "hanner": Hanner-polytope groups evaluated by the cotree support-function
        recursion. "control" / "control_sym": the theory-matched random ternary groups
        (same sizes and per-vertex sparsity; "control_sym" drawn in antipodal pairs).
    features : {"max", "hydra"}, default="max"
        Soft-max and hard-max counts, or Hydra's soft-max and hard-min counts.
    atom_budget : int, default=256
        Atoms (group vertices) per (dilation, input), as in Hydra(k=8, g=64).
    max_vertices : int, default=32
        Largest group (Hanner polytope vertex count).
    kernel_length : int, default=9
        Number of taps.
    p_product : float, default=0.5
        Probability of an l_inf product at each cotree merge.
    contiguous : bool, default=False
        Restrict cotree merges to adjacent tap blocks.
    center : {False, True, "raw", "diff", "mixed"}, default="mixed"
        Patch-mean centering; "mixed" centers every second group.
    increments : bool, default=False
        Evaluate the groups on tap increments (zero-sum integer kernels).
    max_channels : int or "auto", default=8
        Cap on each group's channel subset, ``clip(C // 2, 2, max_channels)``.
    use_diff : bool, default=True
        Also transform the first difference of the series.
    diff_share : float, default=0.5
        Share of the atoms that goes to the first difference.
    batch_size : int, default=64
        Cases per batch of the NumPy backend.
    backend : {"auto", "numpy", "numba"}, default="auto"
        "numba": single-threaded JIT kernels, bit-identical to "numpy" (the reference);
        "auto" uses numba if it is installed.
    random_state : int or None, default=None
        Seed of the dictionaries and channel subsets."""


def _fitted_transform(params: dict, X: np.ndarray) -> HannerHydraTransform:
    """Fit the reference transform and build its Numba layout now, not on the first
    ``transform`` (aeon requires ``transform`` not to change the estimator's state)."""
    tr = HannerHydraTransform(**params).fit(X)
    if _resolve_backend(tr.backend) == "numba":
        tr.__dict__["_packed"] = tr._pack()
    return tr


class _TransformParams:
    """The parameters of ``HannerHydraTransform``, shared by both estimators."""

    def _transform_params(self) -> dict:
        names = HannerHydraTransform._get_param_names()
        return {name: getattr(self, name) for name in names}


class HannerHydraTransformer(_TransformParams, BaseCollectionTransformer):
    __doc__ = f"""Hanner-Hydra transform: competing ternary kernel groups, Hydra-style counts.

    aeon wrapper of :class:`hanner_hydra.HannerHydraTransform` (identical output).

    Parameters
    ----------{_PARAMS_DOC}

    Attributes
    ----------
    transform_ : HannerHydraTransform
        The fitted reference transform.
    n_features_out_ : int
        Number of output features.
    """

    _tags: ClassVar[dict] = {
        "capability:multivariate": True,
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "fit_is_empty": False,
    }

    def __init__(
        self,
        kernel_family: KernelFamily = "hanner",
        features: FeatureSet = "max",
        atom_budget: int = 256,
        max_vertices: int = 32,
        kernel_length: int = 9,
        p_product: float = 0.5,
        contiguous: bool = False,
        center: Centering = "mixed",
        increments: bool = False,
        max_channels: int | Literal["auto"] = 8,
        use_diff: bool = True,
        diff_share: float = 0.5,
        batch_size: int = 64,
        backend: Backend = "auto",
        random_state: int | None = None,
    ) -> None:
        self.kernel_family = kernel_family
        self.features = features
        self.atom_budget = atom_budget
        self.max_vertices = max_vertices
        self.kernel_length = kernel_length
        self.p_product = p_product
        self.contiguous = contiguous
        self.center = center
        self.increments = increments
        self.max_channels = max_channels
        self.use_diff = use_diff
        self.diff_share = diff_share
        self.batch_size = batch_size
        self.backend = backend
        self.random_state = random_state
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HannerHydraTransformer:
        self.transform_ = _fitted_transform(self._transform_params(), X)
        self.n_features_out_ = self.transform_.n_features_out_
        return self

    def _transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.transform_.transform(X)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> list[dict]:
        small = {"atom_budget": 32, "kernel_length": 5, "max_vertices": 10, "random_state": 0}
        return [small, {**small, "kernel_family": "control_sym", "features": "hydra"}]


class HannerHydraClassifier(_TransformParams, BaseClassifier):
    __doc__ = f"""Hanner-Hydra classifier: transform, sparse scaling and a ridge classifier.

    The pipeline of aeon's ``HydraClassifier`` with the Hanner-Hydra transform:
    ``HannerHydraTransform -> SparseScaler -> RidgeClassifierCV(alphas=logspace(-3, 3,
    10))``, identical to :func:`hanner_hydra.make_classifier`.

    ``predict_proba`` returns one-hot rows of the predicted class (aeon's default for a
    classifier without probabilities, as in ``HydraClassifier``). The values are not
    calibrated probabilities.

    Parameters
    ----------{_PARAMS_DOC}
    class_weight : {{None, "balanced"}} or dict, default=None
        Passed to ``RidgeClassifierCV``, as in aeon's ``HydraClassifier``.

    Attributes
    ----------
    classes_ : ndarray
        The class labels.
    n_classes_ : int
        Number of classes.
    pipeline_ : sklearn Pipeline
        The fitted transform, scaler and ridge classifier.
    """

    _tags: ClassVar[dict] = {
        "capability:multivariate": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        kernel_family: KernelFamily = "hanner",
        features: FeatureSet = "max",
        atom_budget: int = 256,
        max_vertices: int = 32,
        kernel_length: int = 9,
        p_product: float = 0.5,
        contiguous: bool = False,
        center: Centering = "mixed",
        increments: bool = False,
        max_channels: int | Literal["auto"] = 8,
        use_diff: bool = True,
        diff_share: float = 0.5,
        batch_size: int = 64,
        backend: Backend = "auto",
        random_state: int | None = None,
        class_weight: str | dict | None = None,
    ) -> None:
        self.kernel_family = kernel_family
        self.features = features
        self.atom_budget = atom_budget
        self.max_vertices = max_vertices
        self.kernel_length = kernel_length
        self.p_product = p_product
        self.contiguous = contiguous
        self.center = center
        self.increments = increments
        self.max_channels = max_channels
        self.use_diff = use_diff
        self.diff_share = diff_share
        self.batch_size = batch_size
        self.backend = backend
        self.random_state = random_state
        self.class_weight = class_weight
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> HannerHydraClassifier:
        self.pipeline_: Pipeline = make_pipeline(
            _fitted_transform(self._transform_params(), X),
            SparseScaler(),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), class_weight=self.class_weight),
        )
        Xt = self.pipeline_[0].transform(X)
        self.pipeline_[1:].fit(Xt, y)
        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline_.predict(X)

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> list[dict]:
        small = {"atom_budget": 32, "kernel_length": 5, "max_vertices": 10, "random_state": 0}
        return [small, {**small, "kernel_family": "control", "class_weight": "balanced"}]
