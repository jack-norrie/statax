import jax
import jax.numpy as jnp

from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType


class BasicBootstrapper(Bootstrapper):
    def ci(self, confidence_level: float = 0.95, alternative: CIType = CIType.TWO_SIDED) -> tuple[jax.Array, jax.Array]:
        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = 2 * self.theta_hat - jnp.quantile(self.bootstrap_replicates, 1 - alpha / 2)
            high = 2 * self.theta_hat - jnp.quantile(self.bootstrap_replicates, alpha / 2)
        elif alternative == CIType.LESS:
            low = jax.Array(-jnp.inf)
            high = 2 * self.theta_hat - jnp.quantile(self.bootstrap_replicates, alpha)
        elif alternative == CIType.GREATER:
            low = 2 * self.theta_hat - jnp.quantile(self.bootstrap_replicates, 1 - alpha)
            high = jax.Array(jnp.inf)
        else:
            raise ValueError(f"Invalid alternative passed, must be of type: {CIType}")

        return (low.astype(float), high.astype(float))
