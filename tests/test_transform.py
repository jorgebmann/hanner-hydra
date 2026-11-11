import numpy as np
import pytest

from hanner_hydra import (
    HannerHydraTransform,
    make_classifier,
    matched_control_group,
    symmetric_control_group,
)
from hanner_hydra.cotree import sample_cotree


def test_control_matches_size_and_sparsity() -> None:
    rng = np.random.default_rng(0)
    tree = sample_cotree(9, rng)
    ctrl = matched_control_group(tree, rng)
    assert ctrl.size == tree.size
    np.testing.assert_array_equal(
        np.sort(np.count_nonzero(ctrl.vertices(), axis=1)),
        np.sort(np.count_nonzero(tree.vertices(), axis=1)),
    )
    assert len({v.tobytes() for v in ctrl.vertices()}) == ctrl.size


@pytest.mark.parametrize("seed", range(10))
def test_symmetric_control_is_centrally_symmetric(seed: int) -> None:
    rng = np.random.default_rng(seed)
    tree = sample_cotree(9, rng)
    ctrl = symmetric_control_group(tree, rng)
    V, a = ctrl.vertices(), tree.antipodes()
    assert ctrl.size == tree.size
    np.testing.assert_array_equal(V[a], -V)  # same antipode permutation as the Hanner group
    np.testing.assert_array_equal(
        np.count_nonzero(V, axis=1), np.count_nonzero(tree.vertices(), axis=1)
    )
    assert len({v.tobytes() for v in V}) == ctrl.size
    # argmin is the antipode of argmax, as for Hanner groups
    X = rng.standard_normal((100, 9))
    _, w_max, w_min = ctrl.max_min(X)
    np.testing.assert_array_equal(w_min, a[w_max])


@pytest.mark.parametrize("family", ["hanner", "control_sym"])
def test_hydra_min_counts_are_permuted_max_counts_for_symmetric_groups(family: str) -> None:
    X = np.random.default_rng(4).standard_normal((4, 60))
    kw = {"kernel_family": family, "atom_budget": 32, "random_state": 0}
    t_max = HannerHydraTransform(features="max", **kw).fit(X)
    F_max = t_max.transform(X)
    F_hyd = HannerHydraTransform(features="hydra", **kw).fit(X).transform(X)
    col = 0
    for key in t_max.groups_:
        for g in t_max.groups_[key]:
            n = g.size
            anti = _antipodes(g.vertices())  # recomputed from the vertices, independently
            np.testing.assert_allclose(F_hyd[:, col : col + n], F_max[:, col : col + n])
            hard_max = F_max[:, col + n : col + 2 * n]
            hard_min = F_hyd[:, col + n : col + 2 * n]
            np.testing.assert_allclose(hard_min[:, anti], hard_max)
            col += 2 * n


def _antipodes(V: np.ndarray) -> np.ndarray:
    index = {v.tobytes(): i for i, v in enumerate(V)}
    return np.array([index[(-v).tobytes()] for v in V])


@pytest.mark.parametrize("family", ["hanner", "control", "control_sym"])
@pytest.mark.parametrize("shape", [(6, 60), (6, 3, 60)])
def test_transform_shapes_and_determinism(family: str, shape: tuple[int, ...]) -> None:
    X = np.random.default_rng(1).standard_normal(shape)
    t = HannerHydraTransform(kernel_family=family, atom_budget=64, random_state=0, batch_size=4)
    F = t.fit_transform(X)
    assert F.shape == (shape[0], t.n_features_out_)
    assert np.isfinite(F).all()
    F2 = HannerHydraTransform(kernel_family=family, atom_budget=64, random_state=0).fit(X)
    np.testing.assert_allclose(F2.transform(X), F)


@pytest.mark.parametrize("features", ["max", "hydra"])
def test_hard_counts_sum_to_series_length(features: str) -> None:
    T = 50
    X = np.random.default_rng(2).standard_normal((3, T))
    t = HannerHydraTransform(features=features, atom_budget=32, use_diff=False, random_state=0)
    t.fit(X)
    F = t.transform(X)
    # Walk the feature layout: per group, [soft-max (size), hard (size)].
    col = 0
    for di in range(len(t.dilations_)):
        for g in t.groups_[di, 0]:
            hard = F[:, col + g.size : col + 2 * g.size]
            np.testing.assert_allclose(hard.sum(axis=1), T)
            col += 2 * g.size
    assert col == F.shape[1]


def test_classifier_separates_easy_problem() -> None:
    rng = np.random.default_rng(3)
    n, T = 40, 80
    t = np.arange(T)
    y = np.repeat([0, 1], n // 2)
    X = rng.standard_normal((n, T)) * 0.3
    X[y == 0] += np.sin(t / 3)
    X[y == 1] += np.sign(np.sin(t / 3))
    clf = make_classifier(atom_budget=64, random_state=0).fit(X[::2], y[::2])
    assert clf.score(X[1::2], y[1::2]) >= 0.9


def test_increments_are_zero_sum_kernels() -> None:
    """Groups on tap increments == support function of the kernels D^T v (zero-sum)."""
    from hanner_hydra import sample_cotree

    rng = np.random.default_rng(11)
    length = 9
    D = np.diff(np.eye(length), axis=0)  # (l - 1, l): (D p)_j = p[j+1] - p[j]
    for _ in range(10):
        tree = sample_cotree(length - 1, rng)
        K = tree.vertices().astype(float) @ D  # kernels in patch coordinates
        np.testing.assert_array_equal(K.sum(axis=1), 0.0)
        assert set(np.unique(K)) <= {-2.0, -1.0, 0.0, 1.0, 2.0}
        P = rng.standard_normal((50, length))
        h, w = tree.support(P[:, 1:] - P[:, :-1])
        resp = P @ K.T
        np.testing.assert_allclose(h, resp.max(axis=1), atol=1e-12)
        np.testing.assert_array_equal(w, resp.argmax(axis=1))


def test_increments_groups_have_one_tap_less() -> None:
    X = np.random.default_rng(0).standard_normal((3, 60))
    t = HannerHydraTransform(increments=True, atom_budget=48, random_state=0).fit(X)
    assert all(g.length == 8 for gs in t.groups_.values() for g in gs)
    assert t.transform(X).shape == (3, t.n_features_out_)


def test_diff_share_moves_atoms_between_inputs() -> None:
    X = np.random.default_rng(0).standard_normal((3, 2, 80))
    t = HannerHydraTransform(diff_share=0.75, atom_budget=100, random_state=0).fit(X)
    for di in range(len(t.dilations_)):
        raw = sum(g.size for g in t.groups_[di, 0])
        diff = sum(g.size for g in t.groups_[di, 1])
        assert 50 <= raw < 50 + 32 and 150 <= diff < 150 + 32
    with pytest.raises(ValueError):
        HannerHydraTransform(diff_share=1.0).fit(X)
    with pytest.raises(ValueError):
        HannerHydraTransform(center="both").fit(X)


def test_max_channels_auto_matches_hydra_channel_draws() -> None:
    X = np.random.default_rng(0).standard_normal((3, 100, 60))
    t = HannerHydraTransform(max_channels="auto", random_state=0).fit(X)
    for (di, inp), chans in t.channels_.items():
        G = len(t.groups_[di, inp])
        assert chans.shape == (G, max(8, -(-256 // G)))
        assert chans.size >= 256  # Hydra: 32 groups x 8 channels
    assert HannerHydraTransform(max_channels=12).fit(X).channels_[0, 0].shape[1] == 12
    with pytest.raises(ValueError):
        HannerHydraTransform(max_channels=1).fit(X)


@pytest.mark.parametrize("n_channels", [1, 3, 17])
def test_max_channels_leaves_low_channel_data_unchanged(n_channels: int) -> None:
    X = np.random.default_rng(1).standard_normal((3, n_channels, 50))
    a = HannerHydraTransform(atom_budget=64, random_state=4).fit(X)
    b = HannerHydraTransform(atom_budget=64, random_state=4, max_channels="auto").fit(X)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))


def test_default_budget_split_is_unchanged() -> None:
    """diff_share=0.5 must draw exactly the same dictionaries as before the option existed."""
    X = np.random.default_rng(0).standard_normal((3, 80))
    a = HannerHydraTransform(atom_budget=64, random_state=3).fit(X)
    b = HannerHydraTransform(atom_budget=64, random_state=3, diff_share=0.5).fit(X)
    np.testing.assert_array_equal(a.transform(X), b.transform(X))
    for di in range(len(a.dilations_)):
        assert 64 <= sum(g.size for g in a.groups_[di, 1]) < 64 + 32


def test_mixed_centering_alternates_groups() -> None:
    """center="mixed": even groups match the uncentered, odd groups the centered transform."""
    X = np.random.default_rng(4).standard_normal((3, 2, 70))
    kw = {"atom_budget": 64, "random_state": 2, "backend": "numpy"}
    mixed = HannerHydraTransform(center="mixed", **kw).fit(X)
    plain = HannerHydraTransform(center=False, **kw).fit(X).transform(X)
    cent = HannerHydraTransform(center=True, **kw).fit(X).transform(X)
    F = mixed.transform(X)
    c = 0
    for di in range(len(mixed.dilations_)):
        for inp in mixed.inputs_:
            for g, group in enumerate(mixed.groups_[di, inp]):
                sl = slice(c, c + 2 * group.size)
                np.testing.assert_array_equal(F[:, sl], (cent if g % 2 else plain)[:, sl])
                c += 2 * group.size
