import jax
import jax.numpy as jnp
import pytest
from jax import random

from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType


class MockBootstrapper(Bootstrapper):
    def ci(self, size: float, alternative: CIType = CIType.TWO_SIDED) -> tuple[jax.Array, jax.Array]:
        return (0.0, 1.0)


@pytest.fixture(scope="function")
def median_bootstrapper():
    return MockBootstrapper(jnp.median)


@pytest.fixture(scope="function")
def mean_bootstrapper():
    return MockBootstrapper(jnp.mean)


class TestBootstrapper:

    def test_data_resampling_shape(self, median_bootstrapper):
        rng_key = random.key(42)
        rng_key, rng_subkey = random.split(rng_key)

        data = jnp.arange(10)
        data_resampled = median_bootstrapper._resample_data(data, rng_subkey)

        assert data.shape == data_resampled.shape

    def test_resampling_shape(self, median_bootstrapper):
        n_resamples = 5
        data = jnp.arange(10)
        median_bootstrapper.resample(data, n_resamples=n_resamples)

        assert len(median_bootstrapper.replicates) == n_resamples

    def test_resampling_variability(self, median_bootstrapper):
        n_resamples = 100
        data = jnp.arange(10)
        median_bootstrapper.resample(data, n_resamples=n_resamples)

        all_equal = True
        first_value = median_bootstrapper.replicates[0]
        for b in range(1, n_resamples):
            bootstrap_replicate = median_bootstrapper.replicates[b]
            if not jnp.allclose(first_value, bootstrap_replicate):
                all_equal = False
                break

        assert not all_equal

    def test_statistic(self, median_bootstrapper):
        n_resamples = 5
        data = jnp.arange(5, 19 + 1)  # [5, 20] -> middle is 12
        median_bootstrapper.resample(data, n_resamples=n_resamples)

        assert median_bootstrapper.theta_hat == 12

    def test_variance(self, mean_bootstrapper):
        rng_key = random.key(42)
        rng_key, rng_subkey = random.split(rng_key)

        n_samples = 1000
        data = random.normal(rng_subkey, shape=(n_samples,))

        n_resamples = 10_000
        mean_bootstrapper.resample(data, n_resamples=n_resamples)

        assert jnp.isclose(mean_bootstrapper.variance(), 1 / n_samples, rtol=0.1)


if __name__ == "__main__":
    pytest.main()
