# Push-2D

![python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![healthcheck](https://github.com/nomutin/push2d-simulator/actions/workflows/healthcheck.yml/badge.svg)](https://github.com/nomutin/push2d-simulator/actions/workflows/healthcheck.yml)
[![lint](https://github.com/nomutin/push2d-simulator/actions/workflows/lint.yml/badge.svg)](https://github.com/nomutin/push2d-simulator/actions/workflows/lint.yml)

![demo](assets/demo.gif)

## Usage

### Install

```shell
poetry add git+https://github.com/nomutin/Push-2D.git
```

### API

```python
from push_2d.core import Push2D
from push_2d.variable import RedAndGreen

env = Push2D(
    space_params=RedAndGreen.SPACE,
    agent_params=RedAndGreen.AGENT,
    obstacles_params=[RedAndGreen.RED, RedAndGreen.GREEN],
)
env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, *_ = env.step(action=action)
```

## References

- [Pymunk](http://www.pymunk.org/en/latest/)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [BringBackShapes Environment](https://github.com/arnavkj1995/BBS)
