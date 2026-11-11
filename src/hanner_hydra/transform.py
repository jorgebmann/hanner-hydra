"""Hydra-style transform with Hanner-polytope (or matched random ternary) kernel groups.

This is the **NumPy reference implementation**: correct and readable, not fast. The
multiplier-free, single-core-optimized kernel (Numba / C++ with SIMD) that the
timing endpoint is measured on is a separate milestone (see ``docs/development-log.md``).

Mirrors Hydra (Dempster et al., 2023) where it can:

* dilations ``2**i`` for ``i = 0 .. floor(log2((T - 1) / (l - 1)))``, zero "same" padding
* two inputs: the series and its first difference
* a fresh dictionary per (dilation, input)
* multivariate: each group sees the sum of a random channel subset of size
  ``clip(C // 2, 2, 8)`` (drawn with replacement; the cap 8 is ``max_channels``)

Features per group (``features``):

* ``"max"`` (default): soft-max counts (sum of the max response, binned by winning atom)
  and hard-max counts (how often each atom wins).
* ``"hydra"``: soft-max counts and Hydra's hard-min counts (how often each atom is the
  minimizer). Kept for ablations only: for a centrally symmetric group (Hanner, or the
  symmetric control) the hard-min count of atom ``j`` equals the hard-max count of its
  antipode, so these features are redundant there but informative for the asymmetric
  control. Decision 2026-11-03, ``docs/development-log.md``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils.validation import check_is_fitted

from hanner_hydra.control import (
    RandomTernaryGroup,
    matched_control_group,
    symmetric_control_group,
)
from hanner_hydra.cotree import Cotree, sample_cotree

KernelFamily = Literal["hanner", "control", "control_sym"]
FeatureSet = Literal["max", "hydra"]
Backend = Literal["auto", "numpy", "numba"]
Centering = bool | Literal["raw", "diff", "mixed"]
Group = Cotree | RandomTernaryGroup

_CONTROLS = {"control": matched_control_group, "control_sym": symmetric_control_group}


class HannerHydraTransform(TransformerMixin, BaseEstimator):
    """Competing ternary kernel groups with Hydra-style counting.

    Parameters
    ----------
    kernel_family : "hanner" (cotree support-function recursion), "control" (random
        ternary groups matched in size and per-vertex sparsity) or "control_sym" (the
        same, drawn in antipodal pairs so the group is centrally symmetric).
    features : "max" (soft-max + hard-max counts, default) or "hydra" (soft-max +
        hard-min counts, as in Hydra; for ablations).
    atom_budget : groups are sampled per (dilation, input) until their total number of
        atoms reaches this budget. Hydra (k=8, g=64) splits its groups between the
        series and its difference, i.e. 8 * 32 = 256 atoms per (dilation, input).
    max_vertices : V_max, the largest allowed group (Hanner polytope vertex count).
    kernel_length : number of taps l.
    p_product : probability of an l_inf product at each allowed cotree merge.
    contiguous : restrict cotree merges to adjacent tap blocks.
    center : subtract each patch's mean before evaluating the group (shift invariance).
        ``True`` / ``False`` for both inputs, ``"raw"`` or ``"diff"`` for one input only,
        or ``"mixed"``: in every block the odd-numbered groups are centered and the even
        ones are not (same dictionaries as center=False), so the ridge can use both.
        Default ``"mixed"``: the sign of the centering effect depends on the data (-0.92
        pts on multivariate, +0.61 on drifting development data); mixed was the most
        robust setting over 34 development datasets (docs/development-log.md, decision log).
    increments : evaluate the groups on the tap increments ``p[j+1] - p[j]`` of each
        patch (groups have ``kernel_length - 1`` taps). This equals kernels ``D^T v``:
        zero-sum, integer entries in {-2..2}, still multiplier-free, and the exact support
        function of a linear image of the Hanner polytope. Level-invariant like
        centering, but with integer kernels. Applied after centering.
    max_channels : cap on the channel subset size of a group, ``clip(C // 2, 2, cap)``.
        Default 8 as in Hydra. ``"auto"`` raises the cap per (dilation, input) to
        ``max(8, ceil(budget / G))`` for G groups, so the groups make at least as many
        channel draws in total as Hydra's ``budget / 8`` groups of 8 channels. Only data
        with ``C >= 18`` is affected; the random stream is unchanged.
    use_diff : also transform the first difference of the series.
    diff_share : share of the two inputs' atoms that goes to the first difference
        (default 0.5: ``atom_budget`` each). The total stays ``2 * atom_budget``.
    batch_size : number of cases processed at once by the NumPy backend (bounds memory).
    backend : "numpy" (reference), "numba" (single-threaded JIT kernels, bit-identical
        output; needs the ``fast`` extra) or "auto" (numba if installed).
    random_state : seed for the dictionaries and channel subsets.
    """

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

    # ------------------------------------------------------------------ fit

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HannerHydraTransform:
        if self.kernel_family != "hanner" and self.kernel_family not in _CONTROLS:
            raise ValueError(f"unknown kernel_family {self.kernel_family!r}")
        if self.features not in ("max", "hydra"):
            raise ValueError(f"unknown features {self.features!r}")
        if self.center not in (False, True, "raw", "diff", "mixed"):
            raise ValueError(f"unknown center {self.center!r}")
        if not 0.0 < self.diff_share < 1.0:
            raise ValueError("diff_share must be in (0, 1)")
        if self.max_channels != "auto" and not (
            isinstance(self.max_channels, int | np.integer) and self.max_channels >= 2
        ):
            raise ValueError("max_channels must be an int >= 2 or 'auto'")
        if self.kernel_length < (3 if self.increments else 2):
            raise ValueError("kernel_length too small")
        X = _as_3d(X)
        _, n_channels, n_timepoints = X.shape
        length = self.kernel_length
        group_length = length - 1 if self.increments else length
        rng = np.random.default_rng(self.random_state)

        max_exp = np.log2((n_timepoints - 1) / (length - 1)) if n_timepoints > length else 0.0
        self.dilations_ = (2 ** np.arange(int(max(max_exp, 0.0)) + 1)).astype(np.int64)
        self.inputs_ = (0, 1) if self.use_diff and n_timepoints > 1 else (0,)
        if len(self.inputs_) == 2:
            budget_diff = round(2 * self.atom_budget * self.diff_share)
            budgets = {0: 2 * self.atom_budget - budget_diff, 1: budget_diff}
        else:
            budgets = {0: self.atom_budget}

        self.groups_: dict[tuple[int, int], list[Group]] = {}
        self.channels_: dict[tuple[int, int], np.ndarray] = {}
        n_features = 0
        for di in range(len(self.dilations_)):
            for inp in self.inputs_:
                groups: list[Group] = []
                atoms = 0
                while atoms < budgets[inp]:
                    tree = sample_cotree(
                        group_length, rng, self.max_vertices, self.p_product, self.contiguous
                    )
                    group: Group = (
                        tree
                        if self.kernel_family == "hanner"
                        else _CONTROLS[self.kernel_family](tree, rng)
                    )
                    groups.append(group)
                    atoms += group.size
                self.groups_[di, inp] = groups
                n_per = self._channels_per_group(n_channels, budgets[inp], len(groups))
                self.channels_[di, inp] = rng.integers(0, n_channels, (len(groups), n_per))
                n_features += 2 * atoms

        self.n_channels_in_ = n_channels
        self.n_features_out_ = n_features
        self.__dict__.pop("_packed", None)  # numba layout cache, rebuilt lazily
        return self

    # ------------------------------------------------------------------ transform

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "groups_")
        X = _as_3d(X)
        if X.shape[1] != self.n_channels_in_:
            raise ValueError(f"expected {self.n_channels_in_} channels, got {X.shape[1]}")
        if _resolve_backend(self.backend) == "numba":
            return self._transform_numba(X)
        n_cases = X.shape[0]
        out = np.zeros((n_cases, self.n_features_out_), np.float64)
        for start in range(0, n_cases, self.batch_size):
            sl = slice(start, min(start + self.batch_size, n_cases))
            out[sl] = self._transform_batch(X[sl])
        return out

    def _transform_batch(self, X: np.ndarray) -> np.ndarray:
        nb = X.shape[0]
        feats: list[np.ndarray] = []
        for di, d in enumerate(self.dilations_):
            for inp in self.inputs_:
                Z = X if inp == 0 else np.diff(X, axis=-1)
                groups = self.groups_[di, inp]
                summed = _channel_sums(Z, self.channels_[di, inp])  # (nb, G, T')
                windows = _dilated_patches(summed, int(d), self.kernel_length)  # (nb,G,T',l)
                for g, group in enumerate(groups):
                    patches = windows[:, g]
                    if _centers_group(self._center_mode(inp), g):
                        patches = _center(patches)
                    if self.increments:
                        patches = patches[..., 1:] - patches[..., :-1]
                    if self.features == "max":
                        h, w_max = group.support(patches)  # each (nb, T')
                        w_hard = w_max  # hard max
                    else:
                        h, w_max, w_hard = group.max_min(patches)  # hard min
                    feats.append(_binned(w_max, group.size, nb, weights=h))  # soft max
                    feats.append(_binned(w_hard, group.size, nb))
        return np.concatenate(feats, axis=1)

    def _transform_numba(self, X: np.ndarray) -> np.ndarray:
        from hanner_hydra import _numba

        packed = self.__dict__.get("_packed")
        if packed is None:
            packed = self._pack()
            self.__dict__["_packed"] = packed
        out = np.zeros((X.shape[0], self.n_features_out_), np.float64)
        hard_min = self.features == "hydra"
        length = self.kernel_length
        for (di, inp), arrays in packed.items():
            Z = X if inp == 0 else np.diff(X, axis=-1)
            Z = np.ascontiguousarray(Z)
            chans = self.channels_[di, inp].astype(np.int64)
            d = int(self.dilations_[di])
            flags = (self._center_mode(inp), self.increments, hard_min)
            if self.kernel_family == "hanner":
                _numba.hanner_block(Z, chans, d, length, *flags, *arrays, out)
            else:
                _numba.ternary_block(Z, chans, d, length, *flags, *arrays, out)
        return out

    def _channels_per_group(self, n_channels: int, budget: int, n_groups: int) -> int:
        if n_channels == 1:
            return 1
        cap = self.max_channels
        if cap == "auto":
            cap = max(8, -(-budget // n_groups))  # ceil(budget / G)
        return int(np.clip(n_channels // 2, 2, cap))

    def _center_mode(self, inp: int) -> int:
        """Centering of input ``inp`` (0: series, 1: difference): 0 none, 1 all, 2 odd groups."""
        if self.center == "mixed":
            return 2
        if isinstance(self.center, str):
            return int(self.center == ("raw", "diff")[inp])
        return int(bool(self.center))

    def _pack(self) -> dict[tuple[int, int], tuple[np.ndarray, ...]]:
        """Flatten each (dilation, input) block's groups into arrays for the kernels.

        Output columns follow the reference layout: blocks in (dilation, input) order,
        and per group ``[soft (size), hard (size)]``.
        """
        packed: dict[tuple[int, int], tuple[np.ndarray, ...]] = {}
        col0 = 0
        for di in range(len(self.dilations_)):
            for inp in self.inputs_:
                groups = self.groups_[di, inp]
                sizes = np.array([g.size for g in groups], np.int64)
                size_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
                col = (col0 + 2 * np.concatenate([[0], np.cumsum(sizes)[:-1]])).astype(np.int64)
                col0 += 2 * int(sizes.sum())
                if self.kernel_family == "hanner":
                    trees: list[Cotree] = groups  # type: ignore[assignment]
                    node_ptr = np.concatenate([[0], np.cumsum([t.n_nodes for t in trees])])
                    packed[di, inp] = (
                        np.concatenate([t.kind for t in trees]).astype(np.int64),
                        np.concatenate([t.left for t in trees]).astype(np.int64),
                        np.concatenate([t.right for t in trees]).astype(np.int64),
                        np.concatenate([t.tap for t in trees]).astype(np.int64),
                        np.concatenate([t.n_vertices for t in trees]).astype(np.int64),
                        node_ptr.astype(np.int64),
                        np.concatenate([t.antipodes() for t in trees]).astype(np.int64),
                        size_ptr,
                        col,
                    )
                else:
                    ctrls: list[RandomTernaryGroup] = groups  # type: ignore[assignment]
                    anti = [
                        c.antipodes_ if c.antipodes_ is not None else np.full(c.size, -1)
                        for c in ctrls
                    ]
                    packed[di, inp] = (
                        np.concatenate([c.vertices() for c in ctrls]).astype(np.int8),
                        np.concatenate(anti).astype(np.int64),
                        size_ptr,
                        col,
                    )
        return packed


class SparseScaler(TransformerMixin, BaseEstimator):
    """Hydra's feature scaler: sqrt, then standardize the non-zero entries."""

    def __init__(self, mask: bool = True, exponent: int = 4) -> None:
        self.mask = mask
        self.exponent = exponent

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> SparseScaler:
        X = np.sqrt(np.clip(X, 0, None))
        self.epsilon_ = (X == 0).mean(axis=0) ** self.exponent + 1e-8
        self.mu_ = X.mean(axis=0)
        # ddof=1 as in aeon's torch implementation (torch.std is unbiased by default)
        self.sigma_ = X.std(axis=0, ddof=1) + self.epsilon_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "mu_")
        X = np.sqrt(np.clip(X, 0, None))
        Z = (X - self.mu_) / self.sigma_
        return Z * (X != 0) if self.mask else Z


def make_classifier(**transform_params: object) -> Pipeline:
    """Hanner-Hydra transform -> SparseScaler -> RidgeClassifierCV (as in Hydra)."""
    return make_pipeline(
        HannerHydraTransform(**transform_params),  # type: ignore[arg-type]
        SparseScaler(),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )


# ---------------------------------------------------------------------- helpers


def _resolve_backend(backend: str) -> str:
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "numpy":
        return "numpy"
    try:
        import numba  # noqa: F401
    except ImportError:
        if backend == "numba":
            raise ImportError("backend='numba' needs numba: uv sync --extra fast") from None
        return "numpy"
    return "numba"


def _as_3d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        return X[:, None, :]
    if X.ndim != 3:
        raise ValueError(f"expected (n_cases, n_timepoints) or (n_cases, C, T), got {X.shape}")
    return X


# The helpers below fix the floating-point summation order (left to right) so that the
# Numba backend reproduces the reference bit for bit. Do not replace them with
# ``.sum()`` / ``.mean()``, whose order (pairwise, SIMD) is an implementation detail.


def _channel_sums(Z: np.ndarray, channels: np.ndarray) -> np.ndarray:
    """``(n, C, T), (G, n_per) -> (n, G, T)``: sum of each group's channels, in order."""
    out = Z[:, channels[:, 0]]
    for k in range(1, channels.shape[1]):
        out = out + Z[:, channels[:, k]]
    return out


def _centers_group(mode: int, g: int) -> bool:
    """Whether group ``g`` of a block with centering ``mode`` (see ``_center_mode``) is centered."""
    return mode == 1 or (mode == 2 and g % 2 == 1)


def _center(patches: np.ndarray) -> np.ndarray:
    """Subtract each patch's mean (taps summed left to right)."""
    acc = patches[..., 0]
    for j in range(1, patches.shape[-1]):
        acc = acc + patches[..., j]
    return patches - (acc / patches.shape[-1])[..., None]


def _dilated_patches(x: np.ndarray, dilation: int, length: int) -> np.ndarray:
    """Zero "same"-padded dilated patches: ``(..., T) -> (..., T, length)`` (a view)."""
    span = (length - 1) * dilation + 1
    pad = ((length - 1) * dilation) // 2
    widths = [(0, 0)] * (x.ndim - 1) + [(pad, span - 1 - pad)]
    xp = np.pad(x, widths)
    return sliding_window_view(xp, span, axis=-1)[..., ::dilation]


def _binned(
    idx: np.ndarray, size: int, n_rows: int, weights: np.ndarray | None = None
) -> np.ndarray:
    """Per-row histogram of ``idx`` (shape ``(n_rows, T)``) into ``size`` bins."""
    flat = (idx + size * np.arange(n_rows)[:, None]).ravel()
    w = None if weights is None else weights.ravel()
    return np.bincount(flat, weights=w, minlength=n_rows * size).reshape(n_rows, size)
