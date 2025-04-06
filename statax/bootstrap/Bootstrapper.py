from abc import ABC, abstractmethod
from typing import Callable
from enum import StrEnum, auto

import jax.numpy as jnp


class CITypes(StrEnum):
    TWO_SIDED = auto()
    LESS = auto()
    GREATER = auto()


class Bootstrapper(ABC):
    def __init__(self, statistic: Callable):
        self._statistic = statistic
        self._bootstrap_replicates = None

    def resample(
        self, data: jnp.ndarray, n_resamples: int = 2000, seed: int = 42
    ) -> None:
        pass

    def variance(self):
        pass

    @abstractmethod
    def ci(self, size: float, alternative: CITypes):
        pass

    def plot_bootstrap_distribution(self) -> None:
        pass
