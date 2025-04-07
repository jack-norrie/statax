from typing import Callable

import jax
import jax.numpy as jnp
from jax import random


class JackKnife:
    def __init__(self, statistic: Callable):
        self._statistic = jax.jit(statistic)
        self._replicates: jax.Array | None = None
        self._mean: jax.Array | None = None

    @property
    def replicates(self) -> jax.Array:
        jackknife_replicates = self._replicates
        if jackknife_replicates is None:
            raise ValueError("JackKnife replicates have not been generated yet. You must call resample() first.")
        return jackknife_replicates

    @property
    def mean(self) -> jax.Array:
        jackknife_mean = self._mean
        if jackknife_mean is None:
            raise ValueError("JackKnife mean has not been generated yet. You must call resample() first.")
        return jackknife_mean

    @staticmethod
    def leave_one_out(data: jax.Array, i: jax.Array) -> jax.Array:
        return jnp.where(jnp.arange(len(data) - 1) < i, data[:-1], data[1:])

    def resample(self, data: jax.Array) -> None:
        n = len(data)
        self._theta_hat = self._statistic(data)

        @jax.vmap
        @jax.jit
        def _generate_jackknife_replicates(i: jax.Array) -> jax.Array:
            # Create a new array by concatenating all elements except the i-th one
            data_loo = self.leave_one_out(data, i)
            theta_loo = self._statistic(data_loo)
            return theta_loo

        self._replicates = _generate_jackknife_replicates(jnp.arange(n))

        self._mean = jnp.mean(self.replicates)

    def std(self):
        return jnp.sqrt(self.variance())

    def variance(self):
        replicates = self.replicates
        n = len(replicates)
        return (n - 1) / n * jnp.sum(jnp.square(replicates - self.mean))

    def skew(self):
        replicates = self.replicates
        return (jnp.sum(jnp.power(replicates - self.mean, 3))) / (
            6 * jnp.power(jnp.sum(jnp.power(replicates - self.mean, 2)), 1.5)
        )
