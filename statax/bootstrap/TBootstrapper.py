from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType

import jax.numpy as jnp


class TBootstrapper(Bootstrapper):
    def ci(self, confidence_level: float = 0.95, alternative: CIType = CIType.TWO_SIDED) -> tuple[float, float]:
        bootstrap_t_statistics = (self.bootstrap_replicates - self.theta_hat) / self.variance()

        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = self.theta_hat - jnp.quantile(bootstrap_t_statistics, 1 - alpha / 2) * self.variance()
            high = self.theta_hat - jnp.quantile(bootstrap_t_statistics, alpha / 2) * self.variance()
        elif alternative == CIType.LESS:
            low = -jnp.inf
            high = self.theta_hat - jnp.quantile(bootstrap_t_statistics, alpha) * self.variance()
        elif alternative == CIType.GREATER:
            low = self.theta_hat - jnp.quantile(bootstrap_t_statistics, 1 - alpha) * self.variance()
            high = jnp.inf
        else:
            raise ValueError(f"Invalid alternative passed, must be of type: {CIType}")

        return (float(low), float(high))
