import numpy as np
import pytest

from hanner_hydra.cotree import cross_polytope, cube, sample_cotree


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("seed", range(20))
def test_support_recursion_matches_brute_force(seed: int, contiguous: bool) -> None:
    rng = np.random.default_rng(seed)
    tree = sample_cotree(9, rng, max_vertices=32, contiguous=contiguous)
    V = tree.vertices().astype(np.float64)
    X = rng.standard_normal((200, 9))

    h, w = tree.support(X)
    responses = X @ V.T
    np.testing.assert_allclose(h, responses.max(axis=1))
    np.testing.assert_array_equal(w, responses.argmax(axis=1))


@pytest.mark.parametrize("seed", range(10))
def test_ties_resolve_like_argmax(seed: int) -> None:
    # Integer patches produce many exact ties; recursion must still match np.argmax.
    rng = np.random.default_rng(seed)
    tree = sample_cotree(9, rng, max_vertices=64)
    X = rng.integers(-2, 3, (500, 9)).astype(np.float64)
    h, w = tree.support(X)
    responses = X @ tree.vertices().T.astype(np.float64)
    np.testing.assert_array_equal(w, responses.argmax(axis=1))
    np.testing.assert_allclose(h, responses.max(axis=1))


@pytest.mark.parametrize("seed", range(20))
def test_vertex_set_properties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    tree = sample_cotree(9, rng, max_vertices=32)
    V = tree.vertices()
    assert V.shape == (tree.size, 9)
    assert 18 <= tree.size <= 32
    assert set(np.unique(V)) <= {-1, 0, 1}
    assert len({v.tobytes() for v in V}) == tree.size  # distinct
    # every tap is used by some vertex
    assert np.all(np.abs(V).sum(axis=0) > 0)


@pytest.mark.parametrize("seed", range(10))
def test_antipodes(seed: int) -> None:
    tree = sample_cotree(9, np.random.default_rng(seed), max_vertices=32)
    V = tree.vertices()
    a = tree.antipodes()
    np.testing.assert_array_equal(V[a], -V)
    np.testing.assert_array_equal(a[a], np.arange(tree.size))


def test_max_min_uses_central_symmetry() -> None:
    rng = np.random.default_rng(0)
    tree = sample_cotree(9, rng)
    X = rng.standard_normal((100, 9))
    _, _, w_min = tree.max_min(X)
    np.testing.assert_array_equal(w_min, (X @ tree.vertices().T.astype(float)).argmin(axis=1))


def test_extreme_polytopes() -> None:
    x = np.array([[0.5, -2.0, 1.0]])
    c = cube(3)
    assert c.size == 8
    h, w = c.support(x)
    np.testing.assert_array_equal(c.vertices()[w[0]], [1, -1, 1])  # sign pattern
    assert h[0] == pytest.approx(3.5)  # l1 norm

    o = cross_polytope(3)
    assert o.size == 6
    h, w = o.support(x)
    np.testing.assert_array_equal(o.vertices()[w[0]], [0, -1, 0])  # signed argmax |x_i|
    assert h[0] == pytest.approx(2.0)  # l_inf norm


def test_sampler_rejects_infeasible_budget() -> None:
    with pytest.raises(ValueError):
        sample_cotree(9, np.random.default_rng(0), max_vertices=17)


def test_sampler_respects_budget_at_cross_polytope_minimum() -> None:
    for seed in range(10):
        tree = sample_cotree(9, np.random.default_rng(seed), max_vertices=18, p_product=1.0)
        assert tree.size == 18
