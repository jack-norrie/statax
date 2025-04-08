import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType
from statax.jackknife import JackKnife

from jax import random


class BCaBootstrapper(Bootstrapper):
    def resample(self, data: jax.Array, n_resamples: int = 2000, key: jax.Array = random.key(42)) -> None:
        super().resample(data, n_resamples, key)

        # Add jackknife resampling such that skew can be estiamted
        jackknife = JackKnife(self._statistic)
        jackknife.resample(data)
        self._jackknife_skew = -jackknife.skew()

    def ci(self, confidence_level: float = 0.95, alternative: CIType = CIType.TWO_SIDED) -> tuple[jax.Array, jax.Array]:
        p0 = jnp.mean(self.bootstrap_replicates <= self.theta_hat)
        z0 = norm.ppf(p0)

        a = self._jackknife_skew

        def percentile_modifier(beta: float):
            zb = norm.ppf(beta)
            return norm.cdf(z0 + (z0 + zb) / (1 - a * (z0 + zb)))

        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = jnp.quantile(self.bootstrap_replicates, percentile_modifier(alpha / 2))
            high = jnp.quantile(self.bootstrap_replicates, percentile_modifier(1 - alpha / 2))
        elif alternative == CIType.LESS:
            low = jax.Array(-jnp.inf)
            high = jnp.quantile(self.bootstrap_replicates, percentile_modifier(1 - alpha))
        elif alternative == CIType.GREATER:
            low = jnp.quantile(self.bootstrap_replicates, percentile_modifier(alpha))
            high = jax.Array(jnp.inf)
        else:
            raise ValueError(f"Invalid alternative passed, must be of type: {CIType}")

        return (low.astype(float), high.astype(float))
