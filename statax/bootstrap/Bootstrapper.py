from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, random

from statax.bootstrap.types import CIType


class Bootstrapper(ABC):
    def __init__(self, statistic: Callable):
        self._statistic = jax.jit(statistic)
        self._bootstrap_replicates: jax.Array | None = None
        self._theta_hat: jax.Array | None = None

    @property
    def theta_hat(self) -> jax.Array:
        theta_hat = self._theta_hat
        if theta_hat is None:
            raise ValueError("Statistic estimate has not been generated yet. You must call resample() first.")
        return theta_hat

    @property
    def bootstrap_replicates(self) -> jax.Array:
        bootstrap_replicates = self._bootstrap_replicates
        if bootstrap_replicates is None:
            raise ValueError("Bootstrap replicates have not been generated yet. You must call resample() first.")
        return bootstrap_replicates

    def _resample_data(self, data: Array, rng_key: jax.Array):
        _, rng_subkey = random.split(rng_key)
        resampled_idxs = random.choice(rng_subkey, jnp.arange(len(data)), shape=(len(data),), replace=True)
        data_resampled = data.at[resampled_idxs].get()
        return data_resampled

    def resample(self, data: jax.Array, n_resamples: int = 2000, key: jax.Array = random.key(42)) -> None:
        key, subkey = random.split(key)

        self._theta_hat = self._statistic(data)

        @jax.vmap
        @jax.jit
        def _generate_bootstrap_replicate(rng_key: jax.Array) -> jax.Array:
            data_resampled = self._resample_data(data, rng_key)
            theta_boot = self._statistic(data_resampled)
            return theta_boot

        self._bootstrap_replicates = _generate_bootstrap_replicate(random.split(subkey, n_resamples))

    def variance(self):
        return jnp.var(self.bootstrap_replicates)

    @abstractmethod
    def ci(self, confidence_level: float = 0.95, alternative: CIType = CIType.TWO_SIDED) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError
