"""Utilities for loading and resolving config files."""

from __future__ import annotations

import math
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def add(*x: int) -> int:
    """Add all the arguments."""
    return sum(x)


def mul(*x: int) -> int:
    """Multiply all the arguments."""
    return math.prod(x)


def sub(x: int, y: int) -> int:
    """Subtract y from x."""
    return x - y


def half(x: int) -> int:
    """Divide x by 2."""
    return x // 2


def load_config(setting_name: str) -> DictConfig:
    """Convert model config `.yaml` to `Dictconfig` with custom resolvers."""
    try:
        OmegaConf.register_new_resolver("add", add)
        OmegaConf.register_new_resolver("mul", mul)
        OmegaConf.register_new_resolver("sub", sub)
        OmegaConf.register_new_resolver("half", half)
    except ValueError:
        pass

    path = Path(__file__).parent.parent / "settings" / f"{setting_name}.yaml"
    config = OmegaConf.load(path)
    OmegaConf.resolve(config)

    if not isinstance(config, DictConfig):
        msg = "ListConfig does not support"
        raise TypeError(msg)

    return config
