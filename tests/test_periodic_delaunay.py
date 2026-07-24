import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astra.desiproc.implement_astra import compute_delaunay_pairs


_TETRA_EDGES = np.asarray(
    ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    dtype=np.intp,
)


def _bounds(periodic_box):
    lower, upper = periodic_box
    lower = np.broadcast_to(np.asarray(lower, dtype=np.float64), (3,))
    upper = np.broadcast_to(np.asarray(upper, dtype=np.float64), (3,))
    return lower, upper


def _reference_27_image_pairs(points, periodic_box):
    """Independent, deliberately expensive periodic reference for small tests."""
    lower, upper = _bounds(periodic_box)
    lengths = upper - lower
    central = np.mod(np.asarray(points, dtype=np.float64) - lower, lengths)
    n_points = len(central)
    shifts = [(0, 0, 0)]
    shifts.extend(
        shift for shift in itertools.product((-1, 0, 1), repeat=3)
        if shift != (0, 0, 0)
    )
    tiled = np.vstack(
        [central + np.asarray(shift) * lengths for shift in shifts]
    )
    tri = Delaunay(tiled)

    # Only the star of the central image is needed. Mapping all simplices near
    # the outer hull would admit finite-tiling boundary artefacts.
    raw_edges = tri.simplices[:, _TETRA_EDGES].reshape(-1, 2)
    raw_edges = raw_edges[np.any(raw_edges < n_points, axis=1)]
    mapped = raw_edges % n_points
    mapped.sort(axis=1)
    mapped = mapped[mapped[:, 0] != mapped[:, 1]]
    return np.unique(mapped.astype(np.int64, copy=False), axis=0)


@pytest.mark.parametrize(
    'periodic_box,n_points,seed',
    [
        ((0.0, 1.0), 180, 13),
        (([-2.0, 3.0, 10.0], [5.0, 14.0, 18.0]), 220, 29),
        ((-5.0, 5.0), 180, 41),
    ],
)
def test_periodic_pairs_match_full_27_image_reference(periodic_box, n_points, seed):
    lower, upper = _bounds(periodic_box)
    points = np.random.default_rng(seed).uniform(lower, upper, size=(n_points, 3))

    actual = compute_delaunay_pairs(points, periodic_box=periodic_box)
    expected = _reference_27_image_pairs(points, periodic_box)

    np.testing.assert_array_equal(actual, expected)


def test_periodic_pairs_are_invariant_under_modular_translation():
    periodic_box = ([-3.0, 2.0, 11.0], [7.0, 15.0, 19.0])
    lower, upper = _bounds(periodic_box)
    lengths = upper - lower
    points = np.random.default_rng(123).uniform(lower, upper, size=(240, 3))
    translated = lower + np.mod(
        points - lower + np.asarray([0.37, -0.22, 1.41]) * lengths,
        lengths,
    )

    original_pairs = compute_delaunay_pairs(points, periodic_box=periodic_box)
    translated_pairs = compute_delaunay_pairs(
        translated, periodic_box=periodic_box
    )

    np.testing.assert_array_equal(original_pairs, translated_pairs)


def test_periodic_graph_connects_points_across_a_box_face():
    rng = np.random.default_rng(7)
    interior = rng.uniform(0.2, 0.8, size=(80, 3))
    points = np.vstack(
        (
            [0.01, 0.50, 0.50],
            [0.99, 0.501, 0.499],
            interior,
        )
    )

    pairs = compute_delaunay_pairs(points, periodic_box=(0.0, 1.0))

    assert np.any(np.all(pairs == (0, 1), axis=1))


def test_periodic_pairs_have_no_duplicates_or_self_edges():
    points = np.random.default_rng(22).random((160, 3))

    pairs = compute_delaunay_pairs(points, periodic_box=(0.0, 1.0))

    assert pairs.dtype == np.int64
    assert np.all(pairs[:, 0] < pairs[:, 1])
    assert len(pairs) == len(np.unique(pairs, axis=0))


@pytest.mark.parametrize(
    'points,periodic_box,error_match',
    [
        (np.ones((4, 2)), (0.0, 1.0), 'shape'),
        (np.ones((3, 3)), (0.0, 1.0), 'at least four'),
        (
            np.asarray(
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4],
                 [0.3, np.nan, 0.5], [0.4, 0.5, 0.6]]
            ),
            (0.0, 1.0),
            'finite',
        ),
        (
            np.asarray(
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4],
                 [0.3, 0.4, 0.5], [1.1, 0.5, 0.6]]
            ),
            (0.0, 1.0),
            'outside',
        ),
        (np.random.default_rng(1).random((8, 3)), (0.0,), 'lower, upper'),
        (np.random.default_rng(2).random((8, 3)), (1.0, 0.0), 'greater'),
        (
            np.random.default_rng(3).random((8, 3)),
            ([0.0, 0.0, np.nan], [1.0, 1.0, 1.0]),
            'finite',
        ),
    ],
)
def test_periodic_validation_errors(points, periodic_box, error_match):
    with pytest.raises(ValueError, match=error_match):
        compute_delaunay_pairs(points, periodic_box=periodic_box)


def test_duplicate_coordinates_after_wrapping_are_rejected():
    points = np.asarray(
        [
            [0.0, 0.2, 0.3],
            [1.0, 0.2, 0.3],
            [0.2, 0.4, 0.5],
            [0.7, 0.8, 0.9],
            [0.3, 0.6, 0.1],
        ]
    )

    with pytest.raises(ValueError, match='duplicate wrapped'):
        compute_delaunay_pairs(points, periodic_box=(0.0, 1.0))


def test_omitting_periodic_box_preserves_nonperiodic_result():
    points = np.random.default_rng(81).random((80, 3))
    tri = Delaunay(points)
    raw_edges = tri.simplices[:, _TETRA_EDGES].reshape(-1, 2)
    raw_edges.sort(axis=1)
    expected = np.unique(raw_edges.astype(np.int64, copy=False), axis=0)

    actual = compute_delaunay_pairs(points)

    np.testing.assert_array_equal(np.unique(actual, axis=0), expected)
