"""Numba backend: single-threaded kernels for one (dilation, input) block.

Each kernel reproduces the NumPy reference (``transform.py``) **bit for bit**:

* channel sums, patch means and ternary dot products are accumulated left to right,
  exactly like ``_channel_sums``, ``_center`` and ``RandomTernaryGroup.responses``
* the Hanner recursion uses the same tie-breaking (leaf: ``x > 0``; l1 sum: ``h_l >= h_r``)
* soft counts are accumulated in time order starting from 0.0, like ``np.bincount``

``fastmath`` must stay off: reassociating sums would break the equivalence test.

The Hanner kernel is multiplier-free in the patch arithmetic (abs, add, compare and
select only). The only multiplications are integer index arithmetic for the winner.

Layout (v2): each group's channel sum is written once into a zero-padded buffer ``sp``,
so tap ``j`` of the patch at time ``t`` is ``sp[t + j * d]``: for a block of ``BLOCK``
consecutive anchors every tap is a contiguous slice and needs no bounds check. Cotree
nodes are evaluated node-major over the block, so the branch on the node kind runs once
per block and the inner loops are branch-free selects that LLVM can vectorize.
"""

from __future__ import annotations

import numpy as np
from numba import njit

BLOCK = 128


@njit(cache=True, nogil=True)
def _fill_padded(Z, i, chans, g, pad, sp):  # pragma: no cover - compiled
    """``sp[pad:pad+T] = sum of the group's channels`` (left to right), zeros elsewhere."""
    n_per = chans.shape[1]
    T = Z.shape[2]
    sp[:] = 0.0
    c0 = chans[g, 0]
    for t in range(T):
        sp[pad + t] = Z[i, c0, t]
    for k in range(1, n_per):
        ck = chans[g, k]
        for t in range(T):
            sp[pad + t] = sp[pad + t] + Z[i, ck, t]


@njit(cache=True, nogil=True)
def _block_patches(sp, t0, nb, d, length, center, mean, P):  # pragma: no cover - compiled
    """``P[j, b]`` = tap ``j`` of the (centered) patch at anchor ``t0 + b``."""
    if center:
        for b in range(nb):
            mean[b] = sp[t0 + b]
        for j in range(1, length):
            off = t0 + j * d
            for b in range(nb):
                mean[b] = mean[b] + sp[off + b]
        for b in range(nb):
            mean[b] = mean[b] / length
        for j in range(length):
            off = t0 + j * d
            for b in range(nb):
                P[j, b] = sp[off + b] - mean[b]
    else:
        for j in range(length):
            off = t0 + j * d
            for b in range(nb):
                P[j, b] = sp[off + b]


@njit(cache=True, nogil=True)
def _to_increments(nb, length, P):  # pragma: no cover - compiled
    """``P[j] = P[j + 1] - P[j]`` for ``j < length - 1``, in place (ascending j is safe)."""
    for j in range(length - 1):
        for b in range(nb):
            P[j, b] = P[j + 1, b] - P[j, b]


@njit(cache=True, nogil=True)
def hanner_block(
    Z,  # (n, C, T) float64
    chans,  # (G, n_per) int64
    d,  # dilation
    length,  # taps of the patch
    center,  # 0: none, 1: all groups, 2: odd-numbered groups (center="mixed")
    increments,  # bool: groups see the length - 1 tap increments
    hard_min,  # bool: hard counts of the minimizer (antipode) instead of the maximizer
    kind,  # packed node arrays, group g occupies node_ptr[g]:node_ptr[g+1]
    left,  # child indices, relative to the group's first node
    right,
    tap,
    nv,
    node_ptr,
    anti,  # packed antipode permutations, group g occupies size_ptr[g]:size_ptr[g+1]
    size_ptr,
    col,  # (G,) first output column of each group
    out,  # (n, F) float64, accumulated in place
):  # pragma: no cover - compiled
    n, _, T = Z.shape
    G = chans.shape[0]
    span = (length - 1) * d + 1
    pad = ((length - 1) * d) // 2
    max_nodes = 0
    for g in range(G):
        max_nodes = max(max_nodes, node_ptr[g + 1] - node_ptr[g])
    H = np.empty((max_nodes, BLOCK), np.float64)
    W = np.empty((max_nodes, BLOCK), np.int64)
    P = np.empty((length, BLOCK), np.float64)
    mean = np.empty(BLOCK, np.float64)
    sp = np.empty(T + span - 1, np.float64)

    for i in range(n):
        for g in range(G):
            _fill_padded(Z, i, chans, g, pad, sp)
            cg = center == 1 or (center == 2 and g % 2 == 1)
            base = node_ptr[g]
            nn = node_ptr[g + 1] - base
            a0 = size_ptr[g]
            size = size_ptr[g + 1] - a0
            c = col[g]
            for t0 in range(0, T, BLOCK):
                nb = min(BLOCK, T - t0)
                _block_patches(sp, t0, nb, d, length, cg, mean, P)
                if increments:
                    _to_increments(nb, length, P)
                for q in range(nn):
                    k = kind[base + q]
                    if k == 0:  # leaf [-1, 1]
                        j = tap[base + q]
                        for b in range(nb):
                            x = P[j, b]
                            H[q, b] = abs(x)
                            W[q, b] = 1 if x > 0 else 0
                    else:
                        lo = left[base + q]
                        ro = right[base + q]
                        if k == 1:  # l1 sum: max, winner of the larger child
                            off = nv[base + lo]
                            for b in range(nb):
                                hl = H[lo, b]
                                hr = H[ro, b]
                                take_left = hl >= hr
                                H[q, b] = hl if take_left else hr
                                W[q, b] = W[lo, b] if take_left else off + W[ro, b]
                        else:  # l_inf product: sum, concatenated winners
                            nr = nv[base + ro]
                            for b in range(nb):
                                H[q, b] = H[lo, b] + H[ro, b]
                                W[q, b] = W[lo, b] * nr + W[ro, b]
                r = nn - 1
                if hard_min:
                    for b in range(nb):
                        w = W[r, b]
                        out[i, c + w] += H[r, b]
                        out[i, c + size + anti[a0 + w]] += 1.0
                else:
                    for b in range(nb):
                        w = W[r, b]
                        out[i, c + w] += H[r, b]
                        out[i, c + size + w] += 1.0


@njit(cache=True, nogil=True)
def ternary_block(
    Z,  # (n, C, T) float64
    chans,  # (G, n_per) int64
    d,
    length,
    center,
    increments,
    hard_min,
    verts,  # (sum sizes, length) int8, group g occupies size_ptr[g]:size_ptr[g+1]
    anti,  # packed antipodes (symmetric groups) or -1 (compute argmin directly)
    size_ptr,
    col,
    out,
):  # pragma: no cover - compiled
    n, _, T = Z.shape
    G = chans.shape[0]
    span = (length - 1) * d + 1
    pad = ((length - 1) * d) // 2
    P = np.empty((length, BLOCK), np.float64)
    mean = np.empty(BLOCK, np.float64)
    acc = np.empty(BLOCK, np.float64)
    best = np.empty(BLOCK, np.float64)
    worst = np.empty(BLOCK, np.float64)
    bi = np.empty(BLOCK, np.int64)
    gl = length - 1 if increments else length  # taps seen by the groups
    wi = np.empty(BLOCK, np.int64)
    sp = np.empty(T + span - 1, np.float64)

    for i in range(n):
        for g in range(G):
            _fill_padded(Z, i, chans, g, pad, sp)
            cg = center == 1 or (center == 2 and g % 2 == 1)
            a0 = size_ptr[g]
            size = size_ptr[g + 1] - a0
            c = col[g]
            symmetric = anti[a0] >= 0
            for t0 in range(0, T, BLOCK):
                nb = min(BLOCK, T - t0)
                _block_patches(sp, t0, nb, d, length, cg, mean, P)
                if increments:
                    _to_increments(nb, length, P)
                for b in range(nb):
                    best[b] = -np.inf
                    worst[b] = np.inf
                    bi[b] = 0
                    wi[b] = 0
                for v in range(size):
                    # Same order as RandomTernaryGroup.responses: p0*v0 + p1*v1 + ...
                    v0 = np.float64(verts[a0 + v, 0])
                    for b in range(nb):
                        acc[b] = P[0, b] * v0
                    for j in range(1, gl):
                        vj = np.float64(verts[a0 + v, j])
                        for b in range(nb):
                            acc[b] = acc[b] + P[j, b] * vj
                    for b in range(nb):
                        a = acc[b]
                        if a > best[b]:
                            best[b] = a
                            bi[b] = v
                        if a < worst[b]:
                            worst[b] = a
                            wi[b] = v
                for b in range(nb):
                    out[i, c + bi[b]] += best[b]
                    if not hard_min:
                        hw = bi[b]
                    elif symmetric:
                        hw = anti[a0 + bi[b]]
                    else:
                        hw = wi[b]
                    out[i, c + size + hw] += 1.0
