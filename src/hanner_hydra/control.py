"""Theory-matched controls: random ternary vertex sets matched to Hanner groups.

For each Hanner group, a control draws a group with the same number of vertices and
the same number of non-zeros per vertex from {-1, 0, 1}^l, and evaluates it by direct
ternary dot products. This isolates the Hanner (cotree) structure from ternarity and
sparsity.

* ``matched_control_group``: vertices drawn independently (the original control).
* ``symmetric_control_group``: vertices drawn in antipodal pairs (u, -u), so the control
  is centrally symmetric like every Hanner polytope (decision 2026-11-03, development-log.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hanner_hydra.cotree import Cotree


@dataclass(frozen=True)
class RandomTernaryGroup:
    """An unstructured ternary dictionary, evaluated in O(|V| * l) per patch."""

    vertices_: np.ndarray  # (size, length) int8
    # Antipode permutation if the group is centrally symmetric (``V[a] == -V``), else None.
    # When set, argmin is taken as the antipode of argmax, exactly as for Hanner groups,
    # so ties (e.g. all-zero padded patches) resolve identically.
    antipodes_: np.ndarray | None = None

    @property
    def length(self) -> int:
        return int(self.vertices_.shape[1])

    @property
    def size(self) -> int:
        return int(self.vertices_.shape[0])

    def vertices(self) -> np.ndarray:
        return self.vertices_

    def responses(self, patches: np.ndarray) -> np.ndarray:
        """``<patch, v>`` for every vertex, shape ``(..., size)``.

        Accumulated tap by tap (not via BLAS) so the summation order is fixed and the
        Numba kernel can reproduce it bit for bit.
        """
        patches = np.asarray(patches, dtype=np.float64)
        V = self.vertices_.astype(np.float64)
        r = patches[..., 0, None] * V[:, 0]
        for j in range(1, self.length):
            r = r + patches[..., j, None] * V[:, j]
        return r

    def support(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = self.responses(patches)
        w = np.argmax(r, axis=-1)
        return np.take_along_axis(r, w[..., None], axis=-1)[..., 0], w

    def antipodes(self) -> np.ndarray:
        if self.antipodes_ is None:
            raise ValueError("group is not centrally symmetric")
        return self.antipodes_

    def max_min(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(h_max, argmax, argmin)``. For a symmetric group argmin is the antipode of
        argmax; otherwise it is computed directly."""
        r = self.responses(patches)
        w_max = np.argmax(r, axis=-1)
        h = np.take_along_axis(r, w_max[..., None], axis=-1)[..., 0]
        if self.antipodes_ is not None:
            return h, w_max, self.antipodes_[w_max]
        return h, w_max, np.argmin(r, axis=-1)


def symmetric_control_group(
    group: Cotree, rng: np.random.Generator, max_tries: int = 1000
) -> RandomTernaryGroup:
    """Centrally symmetric random ternary group matching ``group``'s size and sparsity.

    For every antipodal pair ``(v, -v)`` of the Hanner group, one random ternary vertex
    ``u`` with the same number of non-zeros is drawn and ``(u, -u)`` takes the pair's
    canonical indices. The result has the same size, per-vertex sparsity and antipode
    permutation as ``group``, so its argmin is also the antipode of its argmax. This is
    the second control: it separates the cotree structure from central symmetry.
    """
    ref = group.vertices()
    anti = group.antipodes()
    length = group.length
    nnz = np.count_nonzero(ref, axis=1)
    out = np.zeros_like(ref)
    seen: set[bytes] = set()
    for j in range(group.size):
        if anti[j] < j:
            continue  # filled together with its partner
        k = int(nnz[j])
        for _ in range(max_tries):
            u = np.zeros(length, np.int8)
            pos = rng.choice(length, size=k, replace=False)
            u[pos] = rng.choice(np.array([-1, 1], np.int8), size=k)
            if u.tobytes() not in seen and (-u).tobytes() not in seen:
                break
        else:
            raise RuntimeError(f"could not draw a distinct antipodal pair with {k} non-zeros")
        seen.update((u.tobytes(), (-u).tobytes()))
        out[j], out[anti[j]] = u, -u
    return RandomTernaryGroup(out, antipodes_=anti.copy())


def matched_control_group(
    group: Cotree, rng: np.random.Generator, max_tries: int = 1000
) -> RandomTernaryGroup:
    """Random ternary group matching ``group``'s size and per-vertex sparsity.

    Each vertex keeps its number of non-zeros; positions and signs are redrawn uniformly.
    Duplicate vertices are redrawn (up to ``max_tries`` per vertex) so the dictionary has
    the same number of *distinct* atoms as the Hanner group.
    """
    ref = group.vertices()
    length = group.length
    nnz = np.count_nonzero(ref, axis=1)
    out = np.zeros_like(ref)
    seen: set[bytes] = set()
    for j, k in enumerate(nnz):
        for _ in range(max_tries):
            v = np.zeros(length, np.int8)
            pos = rng.choice(length, size=int(k), replace=False)
            v[pos] = rng.choice(np.array([-1, 1], np.int8), size=int(k))
            key = v.tobytes()
            if key not in seen:
                break
        else:
            raise RuntimeError(f"could not draw a distinct vertex with {k} non-zeros")
        seen.add(key)
        out[j] = v
    return RandomTernaryGroup(out)
