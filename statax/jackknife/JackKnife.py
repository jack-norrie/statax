from typing import Callable

import jax
import jax.numpy as jnp
from jax import random


class JackKnife:
    def __init__(self, statistic: Callable):
        self._statistic = jax.jit(statistic)
        self._jackknife_replicates: jax.Array | None = None
        self._theta_hat: jax.Array | None = None

    @property
    def jackknife_replicates(self) -> jax.Array:
        bootstrap_replicates = self._jackknife_replicates
        if bootstrap_replicates is None:
            raise ValueError("Bootstrap replicates have not been generated yet. You must call resample() first.")
        return bootstrap_replicates

    def resample(self, data: jax.Array) -> None:
        n = len(data)

        @jax.vmap
        @jax.jit
        def _generate_jackknife_replicates(i: jax.Array) -> jax.Array:
            mask = jnp.arange(n) != i
            theta_loo = self._statistic(data[mask])
            return theta_loo

        self._jackknife_replicates = _generate_jackknife_replicates(jnp.arange(n))
