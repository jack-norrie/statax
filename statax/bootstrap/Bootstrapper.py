from abc import ABC, abstractmethod
from typing import Callable
from enum import StrEnum, auto

import jax
from jax import Array, random
import jax.numpy as jnp
from jaxlib.mlir.ir import Value


class CITypes(StrEnum):
    TWO_SIDED = auto()
    LESS = auto()
    GREATER = auto()


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

    def resample(self, data: jax.Array, n_resamples: int = 2000, seed: int = 42) -> None:
        self._theta_hat = self._statistic(data)

        @jax.vmap
        @jax.jit
        def _generate_bootstrap_replicate(rng_key: jax.Array) -> jax.Array:
            data_resampled = self._resample_data(data, rng_key)
            theta_boot = self._statistic(data_resampled)
            return theta_boot

        rng_key = random.key(seed)
        rng_subkeys = random.split(rng_key, n_resamples)
        del rng_key  # good practice.
        self._bootstrap_replicates = _generate_bootstrap_replicate(rng_subkeys)

    def variance(self):
        pass

    @abstractmethod
    def ci(self, size: float, alternative: CITypes) -> tuple[float, float]:
        raise NotImplementedError

    def plot_bootstrap_distribution(self) -> None:
        pass
