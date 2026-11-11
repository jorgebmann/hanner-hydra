"""Labeled cotrees and the Hanner polytopes they define.

A Hanner polytope in R^l is built from l copies of [-1, 1] (one per tap position)
by two binary operations:

* l1 direct sum   K1 (+)_1 K2   = conv((K1 x {0}) u ({0} x K2))   vertices: n1 + n2
* l_inf product   K1 (+)_inf K2 = K1 x K2                         vertices: n1 * n2

A *cotree* records this construction: leaves are tap positions, internal nodes are
labeled ``L1`` or ``LINF``. In the canonical embedding every vertex lies in {-1, 0, 1}^l,
and the support function h(x) = max_v <x, v> follows the tree:

* leaf i          h = |x_i|          winner = sign(x_i)
* LINF (product)  h = h1 + h2        winner = concatenation of both winners
* L1   (sum)      h = max(h1, h2)    winner = winner of the larger child

so the maximum *and* the winning vertex of a whole group cost O(l) additions and
comparisons, versus O(|V| * l) for an unstructured dictionary.

Canonical vertex order (used for the winner index, and matched by ``vertices()``):

* leaf:  index 0 -> -e_i, index 1 -> +e_i
* L1:    left child's vertices first, then the right child's (offset n_left)
* LINF:  index = i_left * n_right + i_right

Ties resolve to the lowest canonical index, which is what ``np.argmax`` over
``vertices()`` returns, so the fast recursion and brute force agree exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEAF: int = 0
L1: int = 1  # l1 direct sum  (vertex counts add)
LINF: int = 2  # l_inf product  (vertex counts multiply)

_KIND_NAMES = {LEAF: "leaf", L1: "l1", LINF: "linf"}


@dataclass(frozen=True)
class Cotree:
    """A labeled cotree over ``length`` tap positions.

    Nodes are stored in topological order (children before parents); the root is the
    last node. Arrays are flat so that the structure can be passed to Numba / C++ later.
    """

    length: int
    kind: np.ndarray  # (n_nodes,) int8: LEAF, L1 or LINF
    left: np.ndarray  # (n_nodes,) int32, -1 for leaves
    right: np.ndarray  # (n_nodes,) int32, -1 for leaves
    tap: np.ndarray  # (n_nodes,) int32, tap position for leaves, -1 otherwise
    n_vertices: np.ndarray  # (n_nodes,) int64, vertex count of each subtree

    @property
    def n_nodes(self) -> int:
        return int(self.kind.shape[0])

    @property
    def root(self) -> int:
        return self.n_nodes - 1

    @property
    def size(self) -> int:
        """Number of vertices of the group's Hanner polytope (= dictionary size)."""
        return int(self.n_vertices[self.root])

    # ------------------------------------------------------------------ geometry

    def vertices(self) -> np.ndarray:
        """All vertices in canonical order, shape ``(size, length)``, dtype int8."""
        verts: list[np.ndarray] = [np.empty((0, self.length), np.int8)] * self.n_nodes
        for i in range(self.n_nodes):
            k = self.kind[i]
            if k == LEAF:
                v = np.zeros((2, self.length), np.int8)
                v[0, self.tap[i]] = -1
                v[1, self.tap[i]] = 1
            elif k == L1:
                v = np.vstack([verts[self.left[i]], verts[self.right[i]]])
            else:
                vl, vr = verts[self.left[i]], verts[self.right[i]]
                v = np.repeat(vl, len(vr), axis=0) + np.tile(vr, (len(vl), 1))
            verts[i] = v
        return verts[self.root]

    def antipodes(self) -> np.ndarray:
        """Permutation ``a`` with ``vertices()[a[j]] == -vertices()[j]``.

        Hanner polytopes are centrally symmetric, so argmin_v <x, v> = antipode(argmax).
        """
        anti: list[np.ndarray] = [np.empty(0, np.int64)] * self.n_nodes
        for i in range(self.n_nodes):
            k = self.kind[i]
            if k == LEAF:
                a = np.array([1, 0], np.int64)
            elif k == L1:
                n_l = self.n_vertices[self.left[i]]
                a = np.concatenate([anti[self.left[i]], n_l + anti[self.right[i]]])
            else:
                n_r = self.n_vertices[self.right[i]]
                a = np.add.outer(anti[self.left[i]] * n_r, anti[self.right[i]]).ravel()
            anti[i] = a
        return anti[self.root]

    def support(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Support function and winning vertex for each row of ``patches``.

        Parameters
        ----------
        patches : array of shape ``(..., length)``

        Returns
        -------
        h : array of shape ``(...)``, ``max_v <patch, v>``
        winner : int64 array of shape ``(...)``, canonical index of the maximizing vertex
        """
        patches = np.asarray(patches)
        if patches.shape[-1] != self.length:
            raise ValueError(f"expected last dim {self.length}, got {patches.shape[-1]}")
        h: list[np.ndarray | None] = [None] * self.n_nodes
        w: list[np.ndarray | None] = [None] * self.n_nodes
        for i in range(self.n_nodes):
            k = self.kind[i]
            if k == LEAF:
                x = patches[..., self.tap[i]]
                h[i] = np.abs(x)
                w[i] = (x > 0).astype(np.int64)
            else:
                lo, ro = self.left[i], self.right[i]
                hl, hr, wl, wr = h[lo], h[ro], w[lo], w[ro]
                if k == L1:
                    take_left = hl >= hr
                    h[i] = np.where(take_left, hl, hr)
                    w[i] = np.where(take_left, wl, self.n_vertices[lo] + wr)
                else:
                    h[i] = hl + hr
                    w[i] = wl * self.n_vertices[ro] + wr
                # Free intermediates as soon as the parent is computed.
                h[lo] = h[ro] = w[lo] = w[ro] = None
        return h[self.root], w[self.root]  # type: ignore[return-value]

    def max_min(self, patches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(h_max, argmax, argmin)`` over the group. By central symmetry the minimum is
        ``-h_max``, attained at the antipode of the maximizer."""
        h, w = self.support(patches)
        return h, w, self._antipodes_cached()[w]

    def _antipodes_cached(self) -> np.ndarray:
        cached = self.__dict__.get("_antipodes")
        if cached is None:
            cached = self.antipodes()
            object.__setattr__(self, "_antipodes", cached)
        return cached

    # ------------------------------------------------------------------ display

    def to_string(self, node: int | None = None) -> str:
        """Infix notation, e.g. ``((x0 x x3) +1 x1)``."""
        i = self.root if node is None else node
        k = self.kind[i]
        if k == LEAF:
            return f"x{self.tap[i]}"
        op = "x" if k == LINF else "+1"
        return f"({self.to_string(self.left[i])} {op} {self.to_string(self.right[i])})"

    def __repr__(self) -> str:
        return f"Cotree(length={self.length}, size={self.size}, {self.to_string()})"


def sample_cotree(
    length: int,
    rng: np.random.Generator,
    max_vertices: int = 32,
    p_product: float = 0.5,
    contiguous: bool = False,
) -> Cotree:
    """Sample a random labeled cotree with at most ``max_vertices`` vertices.

    Starting from ``length`` leaves, two subtrees are repeatedly merged. A merge is an
    l_inf product with probability ``p_product`` whenever the product still allows the
    final polytope to stay within ``max_vertices`` (merging everything else by l1 sums
    gives the smallest final count, ``sum`` of the remaining counts); otherwise it is an
    l1 sum. No rejection sampling is needed.

    Parameters
    ----------
    length : number of taps (kernel length), e.g. 9
    rng : numpy random generator
    max_vertices : V_max; must be >= 2 * length (the cross-polytope)
    p_product : probability of choosing an allowed l_inf product
    contiguous : if True, only merge subtrees that are adjacent in tap order, so every
        subtree covers a contiguous block of taps (a time-local inductive bias)
    """
    if length < 1:
        raise ValueError("length must be >= 1")
    if max_vertices < 2 * length:
        raise ValueError(f"max_vertices must be >= 2 * length = {2 * length}")

    kind = [LEAF] * length
    left = [-1] * length
    right = [-1] * length
    tap = list(range(length))
    nv = [2] * length

    forest = list(range(length)) if contiguous else list(rng.permutation(length))
    total = 2 * length
    while len(forest) > 1:
        if contiguous:
            p = int(rng.integers(len(forest) - 1))
            i, j = p, p + 1
        else:
            i, j = (int(t) for t in rng.choice(len(forest), size=2, replace=False))
        a, b = forest[i], forest[j]
        rest = total - nv[a] - nv[b]
        allow_product = rest + nv[a] * nv[b] <= max_vertices
        k = LINF if allow_product and rng.random() < p_product else L1
        n_new = nv[a] * nv[b] if k == LINF else nv[a] + nv[b]

        kind.append(k)
        left.append(a)
        right.append(b)
        tap.append(-1)
        nv.append(n_new)
        total = rest + n_new

        new_id = len(kind) - 1
        lo, hi = min(i, j), max(i, j)
        forest[lo] = new_id
        del forest[hi]

    return Cotree(
        length=length,
        kind=np.asarray(kind, np.int8),
        left=np.asarray(left, np.int32),
        right=np.asarray(right, np.int32),
        tap=np.asarray(tap, np.int32),
        n_vertices=np.asarray(nv, np.int64),
    )


def cube(length: int) -> Cotree:
    """The cube [-1, 1]^length (all LINF): 2^length vertices, winner = sign pattern."""
    return _chain(length, LINF)


def cross_polytope(length: int) -> Cotree:
    """The cross-polytope (all L1): 2 * length vertices, winner = signed argmax |x_i|."""
    return _chain(length, L1)


def _chain(length: int, op: int) -> Cotree:
    kind, left, right = [LEAF] * length, [-1] * length, [-1] * length
    tap = list(range(length))
    nv = [2] * length
    cur = 0
    for t in range(1, length):
        kind.append(op)
        left.append(cur)
        right.append(t)
        tap.append(-1)
        nv.append(nv[cur] * 2 if op == LINF else nv[cur] + 2)
        cur = len(kind) - 1
    return Cotree(
        length=length,
        kind=np.asarray(kind, np.int8),
        left=np.asarray(left, np.int32),
        right=np.asarray(right, np.int32),
        tap=np.asarray(tap, np.int32),
        n_vertices=np.asarray(nv, np.int64),
    )
