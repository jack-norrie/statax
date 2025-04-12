import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType
from statax.jackknife import JackKnife
from typing import Callable

from jax import random


class BCaBootstrapper(Bootstrapper):
    def __init__(self, statistic: Callable):
        super().__init__(statistic)

        self._acceleration = None
        self._jn_replicates = None

    @property
    def jn_replicates(self) -> jax.Array:
        """
        Get the array of jackknife replicate values.

        Returns:
            Array of bootstrap replicate values.

        Raises:
            ValueError: If resample() has not been called yet.
        """
        jn_replicates = self._jn_replicates
        if jn_replicates is None:
            raise ValueError("Jackknife replicates have not been generated yet. You must call resample() first.")
        return jn_replicates

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
        super().resample(data, n_resamples, key)

        jackknife = JackKnife(self._statistic)
        jackknife.resample(self.replicates)
        self._jn_replicates = jackknife._replicates

    def ci(self, confidence_level: float = 0.95, alternative: CIType = CIType.TWO_SIDED) -> tuple[jax.Array, jax.Array]:
        """
        Compute confidence interval from bootstrap replicates.

        Args:
            confidence_level: The confidence level, typically between 0 and 1.
                             Default is 0.95 for a 95% confidence interval.
            alternative: The type of confidence interval to compute.
                        Options are TWO_SIDED, LOWER, or UPPER.
                        Default is TWO_SIDED.

        Returns:
            A tuple containing the lower and upper bounds of the confidence interval.
        """
        p0 = jnp.mean(self.replicates <= self.theta_hat)
        z0 = norm.ppf(p0)

        jn_replicates = self.jn_replicates
        jn_mean = jnp.mean(jn_replicates)
        a = jnp.sum(jnp.power(jn_mean - jn_replicates, 3)) / (
            6 * jnp.power(jnp.sum(jnp.power(jn_mean - jn_replicates, 2)), 1.5)
        )

        def percentile_modifier(beta: float):
            zb = norm.ppf(beta)
            return norm.cdf(z0 + (z0 + zb) / (1 - a * (z0 + zb)))

        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = jnp.quantile(self.replicates, percentile_modifier(alpha / 2))
            high = jnp.quantile(self.replicates, percentile_modifier(1 - alpha / 2))
        elif alternative == CIType.LESS:
            low = jax.Array(-jnp.inf)
            high = jnp.quantile(self.replicates, percentile_modifier(1 - alpha))
        elif alternative == CIType.GREATER:
            low = jnp.quantile(self.replicates, percentile_modifier(alpha))
            high = jax.Array(jnp.inf)
        else:
            raise ValueError(f"Invalid alternative passed, must be of type: {CIType}")

        return (low.astype(float), high.astype(float))
