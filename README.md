# Statax

Statax is a JAX-based library for statistical computations, providing efficient and GPU-accelerated implementations of common statistical methods. The library leverages JAX's automatic differentiation, vectorization, and just-in-time compilation capabilities to deliver high-performance statistical functions.

## Features

Currently, Statax focuses on bootstrap methods for confidence interval estimation:

- **Multiple Bootstrap Methods**:
  - Basic Bootstrap
  - Percentile Bootstrap
  - Bias-Corrected (BC) Bootstrap
  - Bias-Corrected and Accelerated (BCa) Bootstrap
  - T Bootstrap
  - Standard Bootstrap

>[!TIP]
> For a detailed breakdown of the mathematics and assumptions behind these methods check out this [article](https://jack-norrie.com/Understanding-Bootstrap-Confidence-Intervals/).

- **JAX Integration**:
  - Fully compatible with JAX arrays and transformations
  - Leverages JAX's JIT compilation for performance
  - Supports GPU acceleration

Although, there is certainly scope for this project to expand in the future.

## Installation

### Package

This package is hosted on [PyPI](https://pypi.org/project/statax/) and as such can be installed with most popular package managers. For example, to install this package using `pip`  simply use the command:

```
pip install statax
```

### Development

This project uses [uv](https://docs.astral.sh/uv/) for package management. Therefore, if you would like to contribute to this project please use `uv` to track any changes to the project's dependencies.

```bash
git clone https://github.com/jack-norrie/statax.git
cd statax
uv sync

```

## Requirements

- Python 3.11+
- JAX (with optional CUDA support)

## Quick Start

### Basic Bootstrap Example

```python
import jax
import jax.numpy as jnp
from jax import random
from statax.bootstrap import PercentileBootstrapper

# Generate some data
key = random.key(42)
data = random.normal(key, shape=(100,))

# Define a statistic function
def mean_statistic(x):
    return jnp.mean(x)

# Create a bootstrapper
bootstrapper = PercentileBootstrapper(mean_statistic)

# Generate bootstrap replicates
bootstrapper.resample(data=data, n_resamples=2000, key=random.key(0))

# Calculate confidence interval
ci_low, ci_high = bootstrapper.ci(confidence_level=0.95)
print(f"95% CI: ({ci_low:.4f}, {ci_high:.4f})")
```

### Comparing Different Bootstrap Methods

```python
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt
from statax.bootstrap import (
    BasicBootstrapper,
    PercentileBootstrapper,
    BCBootstrapper,
    BCaBootstrapper,
    TBootstrapper,
    StandardBootstrapper,
)

# Generate skewed data
key = random.key(42)
data = jnp.exp(random.normal(key, shape=(100,)))

# Define statistic
def median_statistic(x):
    return jnp.median(x)

# Compare different bootstrap methods
bootstrappers = {
    "Basic": BasicBootstrapper(median_statistic),
    "Percentile": PercentileBootstrapper(median_statistic),
    "BC": BCBootstrapper(median_statistic),
    "BCa": BCaBootstrapper(median_statistic),
    "T": TBootstrapper(median_statistic),
    "Standard": StandardBootstrapper(median_statistic),
}

results = {}
for name, bootstrapper in bootstrappers.items():
    bootstrapper.resample(data=data, n_resamples=2000, key=random.key(0))
    ci_low, ci_high = bootstrapper.ci(confidence_level=0.95)
    results[name] = (ci_low, ci_high)
    print(f"{name} Bootstrap 95% CI: ({ci_low:.4f}, {ci_high:.4f})")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms of the MIT license.
