from typing import Callable

import jax
import jax.numpy as jnp
from jax import random


class JackKnife:
    """
    Implementation of the jackknife resampling method for statistical inference.

    The jackknife is a resampling technique that systematically leaves out one
    observation at a time from the dataset and recalculates the statistic of interest.
    This allows for estimation of bias and variance of a statistic.

    Attributes:
        _statistic: The statistic function to be applied to resampled data
        _replicates: Array of jackknife replicate values
        _mean: Mean of the jackknife replicates
    """

    def __init__(self, statistic: Callable):
        """
        Initialize a JackKnife resampler with a statistic function.

        Args:
            statistic: A function that computes a statistic on data.
                      Should take a JAX array as input and return a scalar or array.
        """
        self._statistic = jax.jit(statistic)
        self._replicates: jax.Array | None = None
        self._mean: jax.Array | None = None

    @property
    def replicates(self) -> jax.Array:
        """
        Get the array of jackknife replicate values.

        Returns:
            Array of jackknife replicate values.

        Raises:
            ValueError: If resample() has not been called yet.
        """
        jackknife_replicates = self._replicates
        if jackknife_replicates is None:
            raise ValueError("JackKnife replicates have not been generated yet. You must call resample() first.")
        return jackknife_replicates

    @property
    def mean(self) -> jax.Array:
        """
        Get the mean of the jackknife replicates.

        Returns:
            Mean value of the jackknife replicates.

        Raises:
            ValueError: If resample() has not been called yet.
        """
        jackknife_mean = self._mean
        if jackknife_mean is None:
            raise ValueError("JackKnife mean has not been generated yet. You must call resample() first.")
        return jackknife_mean

    @staticmethod
    def leave_one_out(data: jax.Array, i: jax.Array) -> jax.Array:
        """
        Create a leave-one-out sample by removing the i-th element from the data.

        Args:
            data: The original data array.
            i: Index of the element to leave out.

        Returns:
            A new array with the i-th element removed.
        """
        return jnp.where(jnp.arange(len(data) - 1) < i, data[:-1], data[1:])

    def resample(self, data: jax.Array) -> None:
        """
        Generate jackknife replicates by systematically leaving out each observation.

        This method computes the statistic on the original data and generates
        jackknife replicates by removing one observation at a time and computing
        the statistic on each reduced dataset.

        Args:
            data: The original data array.
        """
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

    def std(self) -> jax.Array:
        """
        Compute the jackknife estimate of the standard error.

        Returns:
            The jackknife standard error estimate.
        """
        return jnp.sqrt(self.variance())

    def variance(self) -> jax.Array:
        """
        Compute the jackknife estimate of variance.

        Returns:
            The jackknife variance estimate.
        """
        replicates = self.replicates
        n = len(replicates)
        return (n - 1) / n * jnp.sum(jnp.square(replicates - self.mean))
