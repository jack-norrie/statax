import pytest
import jax.numpy as jnp
import jax
from statax.jackknife.JackKnife import JackKnife


@jax.jit
def rmse(data):
    return jnp.sqrt(jnp.mean(jnp.square(data)))


@pytest.fixture(scope="function")
def jackknife():
    return JackKnife(jnp.sqrt)


class TestJackKnife:
    def test_replicates_size(self, jackknife):
        n = 10
        data = jnp.arange(n)

        jackknife.resample(data)

        assert len(jackknife.replicates) == n

    def test_replicates_values(self, jackknife):
        n = 3
        data = jnp.arange(n)

        actual = jackknife.leave_one_out(data, 0)
        expected = jnp.array([1, 2])
        assert jnp.allclose(actual, expected)

        actual = jackknife.leave_one_out(data, 1)
        expected = jnp.array([0, 2])
        assert jnp.allclose(actual, expected)

        actual = jackknife.leave_one_out(data, 2)
        expected = jnp.array([0, 1])
        assert jnp.allclose(actual, expected)


if __name__ == "__main__":
    pytest.main()
