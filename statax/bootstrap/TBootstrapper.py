import jax
import jax.numpy as jnp

from statax.bootstrap.Bootstrapper import Bootstrapper
from statax.bootstrap.types import CIType


class TBootstrapper(Bootstrapper):
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
        bootstrap_t_statistics = (self.replicates - self.theta_hat) / jnp.sqrt(self.variance())

        alpha = 1 - confidence_level
        if alternative == CIType.TWO_SIDED:
            low = self.theta_hat - jnp.quantile(bootstrap_t_statistics, 1 - alpha / 2) * jnp.sqrt(self.variance())
            high = self.theta_hat - jnp.quantile(bootstrap_t_statistics, alpha / 2) * jnp.sqrt(self.variance())
        elif alternative == CIType.LESS:
            low = jnp.array(-jnp.inf)
            high = self.theta_hat - jnp.quantile(bootstrap_t_statistics, alpha) * jnp.sqrt(self.variance())
        elif alternative == CIType.GREATER:
            low = self.theta_hat - jnp.quantile(bootstrap_t_statistics, 1 - alpha) * jnp.sqrt(self.variance())
            high = jnp.array(jnp.inf)
        else:
            raise ValueError(f"Invalid alternative passed, must be of type: {CIType}")

        return (low.astype(float), high.astype(float))
