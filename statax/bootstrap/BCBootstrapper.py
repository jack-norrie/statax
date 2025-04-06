from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType

import jax.numpy as jnp
from jax.scipy.stats import norm


class TBootstrapper(Bootstrapper):
    def ci(self, confidence_level: float, alternative: CIType) -> tuple[float, float]:
        p0 = jnp.mean(self.bootstrap_replicates < self.theta_hat)
        z0 = norm.ppf(p0)

        def percentile_modifier(beta: float):
            return norm.cdf(2 * z0 + norm.ppf(beta))

        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = jnp.quantile(self.bootstrap_replicates, percentile_modifier(alpha / 2))
            high = jnp.quantile(self.bootstrap_replicates, percentile_modifier(1 - alpha / 2))
        elif alternative == CIType.LESS:
            low = -jnp.inf
            high = jnp.quantile(self.bootstrap_replicates, percentile_modifier(1 - alpha))
        elif alternative == CIType.GREATER:
            low = jnp.quantile(self.bootstrap_replicates, percentile_modifier(alpha))
            high = jnp.inf
        else:
            raise ValueError(f"Invalid alternaive passed, must be of type: {CIType}")

        return (float(low), float(high))
