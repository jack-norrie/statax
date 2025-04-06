from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType

import jax.numpy as jnp
from jax.scipy.stats import norm


class TBootstrapper(Bootstrapper):
    def ci(self, confidence_level: float, alternative: CIType) -> tuple[float, float]:
        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = jnp.quantile(self.bootstrap_replicates, alpha / 2)
            high = jnp.quantile(self.bootstrap_replicates, 1 - alpha / 2)
        elif alternative == CIType.LESS:
            low = -jnp.inf
            high = jnp.quantile(self.bootstrap_replicates, 1 - alpha)
        elif alternative == CIType.GREATER:
            low = jnp.quantile(self.bootstrap_replicates, alpha)
            high = jnp.inf
        else:
            raise ValueError(f"Invalid alternaive passed, must be of type: {CIType}")

        return (float(low), float(high))
