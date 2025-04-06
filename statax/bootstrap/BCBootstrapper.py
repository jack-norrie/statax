from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType

import jax.numpy as jnp
from jax.scipy.stats import norm


class BCBootstrapper(Bootstrapper):
    def ci(self, confidence_level: float, alternative: CIType) -> tuple[float, float]:
        p0 = jnp.mean(self.bootstrap_replicates < self.theta_hat)
        z0 = norm.ppf(p0)

        a = 1

        def percentile_modifier(beta: float):
            zb = norm.ppf(beta)
            return norm.cdf(z0 + (z0 + zb) / (1 - a * (z0 + zb)))

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
            raise ValueError(f"Invalid alternative passed, must be of type: {CIType}")

        return (float(low), float(high))
