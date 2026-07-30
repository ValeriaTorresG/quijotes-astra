from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fisher import validate_fisher_stability as validation


def test_sample_covariance_matches_numpy():
    data = np.asarray(
        [
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 1.0],
            [5.0, 3.0, 7.0],
            [8.0, 9.0, 2.0],
        ]
    )

    actual = validation.sample_covariance(data)

    np.testing.assert_allclose(actual, np.cov(data, rowvar=False, ddof=1))
    np.testing.assert_allclose(actual, actual.T)


def test_oas_correlation_shrinkage_preserves_sample_variances():
    rng = np.random.default_rng(431)
    latent = rng.normal(size=(30, 2))
    data = np.column_stack(
        [
            latent[:, 0],
            2.0 * latent[:, 0] + 0.2 * latent[:, 1],
            latent[:, 1],
            latent[:, 0] - latent[:, 1],
        ]
    )
    sample = validation.sample_covariance(data)

    shrunk, intensity = validation.oas_correlation_shrinkage(data)

    assert 0.0 <= intensity <= 1.0
    np.testing.assert_allclose(np.diag(shrunk), np.diag(sample))
    np.testing.assert_allclose(shrunk, shrunk.T)
    assert np.linalg.eigvalsh(shrunk)[0] > 0.0


def test_parameter_covariance_matches_inverse_and_rejects_singular():
    fisher = np.asarray([[4.0, 1.0], [1.0, 3.0]])
    actual = validation.parameter_covariance(fisher, context="test")
    np.testing.assert_allclose(actual, np.linalg.inv(fisher))

    with pytest.raises(validation.ValidationError, match="positive definite"):
        validation.parameter_covariance(
            np.asarray([[1.0, 1.0], [1.0, 1.0]]),
            context="singular",
        )


def test_safe_ratios_mask_small_denominators():
    numerator = np.asarray([2.0, 3.0, 4.0])
    denominator = np.asarray([1.0, 0.01, -2.0])

    ratios, valid = validation._safe_ratios(
        numerator,
        denominator,
        floor=0.05,
    )

    np.testing.assert_array_equal(valid, [True, False, True])
    np.testing.assert_allclose(ratios[valid], [2.0, -2.0])
    assert np.isnan(ratios[1])
