from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType

import jax.numpy as jnp


class TBootstrapper(Bootstrapper):
    def ci(self, confidence_level: float, alternative: CIType) -> tuple[float, float]:
        bootstrap_t_statistics = (self.bootstrap_replicates - self.theta_hat) / self.variance()

        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = 2 * self.theta_hat - jnp.quantile(bootstrap_t_statistics, 1 - alpha / 2)
            high = 2 * self.theta_hat - jnp.quantile(bootstrap_t_statistics, alpha / 2)
        elif alternative == CIType.LESS:
            low = -jnp.inf
            high = 2 * self.theta_hat - jnp.quantile(bootstrap_t_statistics, alpha)
        elif alternative == CIType.GREATER:
            low = 2 * self.theta_hat - jnp.quantile(bootstrap_t_statistics, 1 - alpha)
            high = jnp.inf
        else:
            raise ValueError(f"Invalid alternaive passed, must be of type: {CIType}")

        return (float(low), float(high))
