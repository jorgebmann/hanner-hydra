"""Bayesian signed-rank test (Benavoli, Corani, Demšar & Zaffalon, JMLR 2017).

Used for the primary endpoint (``docs/preregistration.md`` §4): per-dataset
accuracies of a candidate ``a`` and a comparator ``b``, in percentage points, with a
region of practical equivalence (ROPE) ``[-rope, rope]``.

The posterior over the three region probabilities ``theta = (left, rope, right)`` is
sampled with a Dirichlet process prior that puts weight ``prior`` (default 0.5, as in
``baycomp``) on one pseudo-observation at ``z0 = 0``. For one draw ``w`` of Dirichlet
weights over ``(z0, z1, ..., zn)``:

    theta_left  = sum_ij w_i w_j I(z_i + z_j < -2 rope)
    theta_right = sum_ij w_i w_j I(z_i + z_j >  2 rope)

with ties at the boundary counted half (the Walsh averages ``(z_i + z_j) / 2`` of the
signed-rank statistic). This follows the reference implementation in ``baycomp``.

Orientation: ``z = a - b``, so ``left`` means *a is practically worse* and ``right`` means
*a is practically better*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignedRankResult:
    """Posterior summary of the Bayesian signed-rank test.

    ``p_left``, ``p_rope`` and ``p_right`` are the posterior probabilities that the
    respective region is the most probable one (fractions of Monte Carlo draws, as
    reported by Benavoli et al. 2017 and ``baycomp``). ``mean_theta`` is the posterior
    mean of ``(theta_left, theta_rope, theta_right)``.
    """

    p_left: float
    p_rope: float
    p_right: float
    mean_theta: tuple[float, float, float]
    n: int
    rope: float

    @property
    def p_not_worse(self) -> float:
        """``P(delta >= -rope)``: the *not worse* criterion (threshold 0.95)."""
        return self.p_rope + self.p_right


def signed_rank_samples(
    a: np.ndarray,
    b: np.ndarray,
    rope: float = 0.5,
    prior: float = 0.5,
    n_samples: int = 50_000,
    random_state: int | np.random.Generator | None = None,
    chunk: int = 5_000,
) -> np.ndarray:
    """Posterior draws of ``(theta_left, theta_rope, theta_right)``, shape ``(n_samples, 3)``."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-d arrays of equal length")
    if rope < 0 or prior <= 0:
        raise ValueError("rope must be >= 0 and prior > 0")
    rng = np.random.default_rng(random_state)

    z = np.concatenate(([0.0], a - b))
    s = z[:, None] + z[None, :]
    below = (s < -2 * rope) + 0.5 * (s == -2 * rope)
    above = (s > 2 * rope) + 0.5 * (s == 2 * rope)
    alpha = np.ones(z.size)
    alpha[0] = prior

    out = np.empty((n_samples, 3))
    for i in range(0, n_samples, chunk):
        w = rng.dirichlet(alpha, min(chunk, n_samples - i))
        left = np.einsum("si,ij,sj->s", w, below, w)
        right = np.einsum("si,ij,sj->s", w, above, w)
        out[i : i + w.shape[0]] = np.column_stack((left, 1.0 - left - right, right))
    return out


def bayesian_signed_rank(
    a: np.ndarray,
    b: np.ndarray,
    rope: float = 0.5,
    prior: float = 0.5,
    n_samples: int = 50_000,
    random_state: int | np.random.Generator | None = 0,
) -> SignedRankResult:
    """Bayesian signed-rank test of ``a`` versus ``b`` (paired per-dataset scores).

    ``a`` and ``b`` must be on the scale of ``rope`` (here: accuracy in percentage points,
    ``rope=0.5``). The seed is fixed by default so reported probabilities are reproducible.
    """
    samples = signed_rank_samples(a, b, rope, prior, n_samples, random_state)
    winner = np.bincount(samples.argmax(axis=1), minlength=3) / samples.shape[0]
    m = samples.mean(axis=0)
    return SignedRankResult(
        p_left=float(winner[0]),
        p_rope=float(winner[1]),
        p_right=float(winner[2]),
        mean_theta=(float(m[0]), float(m[1]), float(m[2])),
        n=int(np.asarray(a).size),
        rope=rope,
    )
