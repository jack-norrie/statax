from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType

import jax.numpy as jnp
from jax.scipy.stats import norm


class StandardBootstrapper(Bootstrapper):
    def ci(self, confidence_level: float, alternative: CIType) -> tuple[float, float]:
        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = self.theta_hat + norm.ppf(alpha / 2) * self.variance()
            high = self.theta_hat + norm.pdf(1 - alpha / 2) * self.variance()
        elif alternative == CIType.LESS:
            low = -jnp.inf
            high = self.theta_hat + norm.pdf(1 - alpha) * self.variance()
        elif alternative == CIType.GREATER:
            low = self.theta_hat + norm.ppf(alpha) * self.variance()
            high = jnp.inf
        else:
            raise ValueError(f"Invalid alternaive passed, must be of type: {CIType}")

        return (float(low), float(high))
