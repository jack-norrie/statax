from abc import ABC, abstractmethod
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array, random

from statax.bootstrap.types import CIType


class Bootstrapper(ABC):
    """
    Abstract base class for bootstrap resampling methods.

    Specific bootstrap methods should inherit from this class and implement
    the confidence interval calculation method.
    """

    def __init__(self, statistic: Callable):
        """
        Initialize a Bootstrapper with a statistic function.

        Args:
            statistic: A function that computes a statistic on data.
                       Should take a JAX array as input and return a scalar or array.
        """
        self._statistic = jax.jit(statistic)
        self._bootstrap_replicates: jax.Array | None = None
        self._theta_hat: jax.Array | None = None

    @property
    def theta_hat(self) -> jax.Array:
        """
        Get the statistic computed on the original data.

        Returns:
            The value of the statistic computed on the original data.

        Raises:
            ValueError: If resample() has not been called yet.
        """
        theta_hat = self._theta_hat
        if theta_hat is None:
            raise ValueError("Statistic estimate has not been generated yet. You must call resample() first.")
        return theta_hat

    @property
    def replicates(self) -> jax.Array:
        """
        Get the array of bootstrap replicate values.

        Returns:
            Array of bootstrap replicate values.

        Raises:
            ValueError: If resample() has not been called yet.
        """
        bootstrap_replicates = self._bootstrap_replicates
        if bootstrap_replicates is None:
            raise ValueError("Bootstrap replicates have not been generated yet. You must call resample() first.")
        return bootstrap_replicates

    def _resample_data(self, data: Array, rng_key: jax.Array):
        """
        Resample data with replacement for bootstrap.

        Args:
            data: The original data array.
            rng_key: JAX random key for reproducibility.

        Returns:
            A resampled version of the data with the same shape.
        """
        _, rng_subkey = random.split(rng_key)
        resampled_idxs = random.choice(rng_subkey, jnp.arange(len(data)), shape=(len(data),), replace=True)
        data_resampled = data.at[resampled_idxs].get()
        return data_resampled

    def resample(self, data: jax.Array, n_resamples: int = 2000, key: jax.Array = random.key(42)) -> None:
        """
        Generate bootstrap replicates by resampling the data.

        This method computes the statistic on the original data and generates
        bootstrap replicates by repeatedly resampling the data with replacement
        and computing the statistic on each resample.

        Args:
            data: The original data array.
            n_resamples: Number of bootstrap resamples to generate. Default is 2000.
            key: JAX random key for reproducibility. Default is a fixed seed.

        Returns:
            None. The results are stored in the object's properties.
        """
        key, subkey = random.split(key)

        self._theta_hat = self._statistic(data)

        @jax.vmap
        @jax.jit
        def _generate_bootstrap_replicate(rng_key: jax.Array) -> jax.Array:
            data_resampled = self._resample_data(data, rng_key)
            theta_boot = self._statistic(data_resampled)
            return theta_boot

        self._bootstrap_replicates = _generate_bootstrap_replicate(random.split(subkey, n_resamples))

    @abstractmethod
    def ci(self, confidence_level: float = 0.95, alternative: CIType = CIType.TWO_SIDED) -> tuple[jax.Array, jax.Array]:
        """
        Compute confidence interval from bootstrap replicates.

        This is an abstract method that must be implemented by subclasses
        to provide specific confidence interval calculation methods.

        Args:
            confidence_level: The confidence level, typically between 0 and 1.
                             Default is 0.95 for a 95% confidence interval.
            alternative: The type of confidence interval to compute.
                        Options are TWO_SIDED, LOWER, or UPPER.
                        Default is TWO_SIDED.

        Returns:
            A tuple containing the lower and upper bounds of the confidence interval.
        """
        raise NotImplementedError

    def variance(self) -> jax.Array:
        """
        Compute the variance of the bootstrap distribution.

        Returns:
            The variance of the bootstrap replicates.

        Raises:
            ValueError: If resample() has not been called yet.
        """
        return jnp.var(self.replicates)
