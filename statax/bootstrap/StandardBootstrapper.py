import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType


class StandardBootstrapper(Bootstrapper):
    def ci(self, confidence_level: float = 0.95, alternative: CIType = CIType.TWO_SIDED) -> tuple[jax.Array, jax.Array]:
        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = self.theta_hat + norm.ppf(alpha / 2) * jnp.sqrt(self.variance())
            high = self.theta_hat + norm.ppf(1 - alpha / 2) * jnp.sqrt(self.variance())
        elif alternative == CIType.LESS:
            low = jax.Array(-jnp.inf)
            high = self.theta_hat + norm.ppf(1 - alpha) * jnp.sqrt(self.variance())
        elif alternative == CIType.GREATER:
            low = self.theta_hat + norm.ppf(alpha) * jnp.sqrt(self.variance())
            high = jax.Array(jnp.inf)
        else:
            raise ValueError(f"Invalid alternative passed, must be of type: {CIType}")

        return (low.astype(float), high.astype(float))
