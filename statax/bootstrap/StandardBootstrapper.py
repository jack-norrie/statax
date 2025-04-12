import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType


class StandardBootstrapper(Bootstrapper):
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
