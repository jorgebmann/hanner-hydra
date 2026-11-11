import numpy as np
import pytest

from hanner_hydra.stats import bayesian_signed_rank, signed_rank_samples


def _brute_force(a, b, rope, w):
    """Literal double sum of Benavoli et al. (2017) for one weight vector."""
    z = np.concatenate(([0.0], np.asarray(a) - np.asarray(b)))
    left = right = 0.0
    for i in range(z.size):
        for j in range(z.size):
            s = z[i] + z[j]
            left += w[i] * w[j] * ((s < -2 * rope) + 0.5 * (s == -2 * rope))
            right += w[i] * w[j] * ((s > 2 * rope) + 0.5 * (s == 2 * rope))
    return left, 1 - left - right, right


def test_samples_match_double_sum() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 9)
    b = rng.normal(0, 1, 9)
    a[3] = b[3] + 0.5  # exact boundary case z_i + z_0 = 2 * rope
    samples = signed_rank_samples(a, b, rope=0.25, n_samples=7, random_state=4, chunk=3)
    w = np.random.default_rng(4)
    alpha = np.ones(10)
    alpha[0] = 0.5  # default prior strength on the pseudo-observation z0 = 0
    draws = [w.dirichlet(alpha, n) for n in (3, 3, 1)]
    for k, wk in enumerate(np.concatenate(draws)):
        np.testing.assert_allclose(samples[k], _brute_force(a, b, 0.25, wk), atol=1e-12)


def test_samples_are_probabilities() -> None:
    rng = np.random.default_rng(1)
    s = signed_rank_samples(rng.normal(size=30), rng.normal(size=30), n_samples=2000)
    assert np.all(s >= -1e-12)
    np.testing.assert_allclose(s.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    ("shift", "expected"),
    [(5.0, "p_right"), (-5.0, "p_left"), (0.0, "p_rope")],
)
def test_clear_cases(shift: float, expected: str) -> None:
    rng = np.random.default_rng(2)
    b = rng.uniform(70, 90, 40)
    a = b + shift + rng.normal(0, 0.05, 40)
    res = bayesian_signed_rank(a, b, rope=0.5)
    assert getattr(res, expected) > 0.99
    assert res.n == 40


def test_not_worse_and_symmetry() -> None:
    rng = np.random.default_rng(3)
    b = rng.uniform(70, 90, 50)
    a = b + rng.normal(0.2, 1.0, 50)
    ab = bayesian_signed_rank(a, b)
    ba = bayesian_signed_rank(b, a)
    assert ab.p_not_worse == pytest.approx(ab.p_rope + ab.p_right)
    # Swapping the arms mirrors the regions (same Dirichlet draws for a fixed seed).
    assert ab.p_left == pytest.approx(ba.p_right)
    assert ab.p_right == pytest.approx(ba.p_left)


def test_deterministic_and_validates() -> None:
    a, b = np.arange(10.0), np.arange(10.0)[::-1]
    assert bayesian_signed_rank(a, b) == bayesian_signed_rank(a, b)
    with pytest.raises(ValueError):
        signed_rank_samples(a, b[:5])
    with pytest.raises(ValueError):
        signed_rank_samples(a, b, rope=-1)
