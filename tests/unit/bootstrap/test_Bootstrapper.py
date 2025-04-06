import jax.numpy as jnp
from jax import random
import pytest

from statax.bootstrap.Bootstrapper import Bootstrapper, CITypes


class MockBootstrapper(Bootstrapper):
    def ci(self, size: float, alternative: CITypes = CITypes.TWO_SIDED) -> tuple[float, float]:
        return (0.0, 1.0)


@pytest.fixture(scope="function")
def median_bootstrapper():
    return MockBootstrapper(jnp.median)


class TestBootstrapper:

    def test_data_resampling(self, median_bootstrapper):
        rng_key = random.key(42)
        rng_key, rng_subkey = random.split(rng_key)

        data = jnp.arange(10)
        data_resampled = median_bootstrapper._resample_data(data, rng_key)

        assert data.shape == data_resampled.shape

    def test_resampling(self, median_bootstrapper):
        n_resamples = 5
        data = jnp.arange(10)
        median_bootstrapper.resample(data, n_resamples=n_resamples)

        assert len(median_bootstrapper.bootstrap_replicates) == n_resamples

    def test_resampling_variability(self, median_bootstrapper):
        n_resamples = 100
        data = jnp.arange(10)
        median_bootstrapper.resample(data, n_resamples=n_resamples)

        all_equal = True
        first_value = median_bootstrapper.bootstrap_replicates[0]
        for b in range(1, n_resamples):
            bootstrap_replicate = median_bootstrapper.bootstrap_replicates[b]
            if not jnp.allclose(first_value, bootstrap_replicate):
                all_equal = False
                break

        assert not all_equal

    def test_statistic(self, median_bootstrapper):
        n_resamples = 5
        data = jnp.arange(5, 19 + 1)  # [5, 20] -> middle is 12
        median_bootstrapper.resample(data, n_resamples=n_resamples)

        assert median_bootstrapper.theta_hat == 12


if __name__ == "__main__":
    pytest.main()
