# Push2D-Simulator

![platform](https://img.shields.io/badge/platform-X11-blue)
![python](https://img.shields.io/badge/python-3.8%20|%203.9-blue)
[![black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

![demo](assets/demo.gif)

## Usage

### Install

```shell
poetry add git+https://github.com/nomutin/push2d-simulator.git
```

### Collect Data

```python
from push2d_simulator import Simulator

simulator = Simulator(config)
simulator.follow()
```

### Action Test

```python
from push2d_simulator import Simulator

simulator = Simulator(config)
for action in action_sequence:
    simulator.replay(action)
    current_observation = simulator.get_observation()
    current_action = simulator.get_action()
```

## References

- [Pymunk](http://www.pymunk.org/en/latest/)
- [BringBackShapes Environment](https://github.com/arnavkj1995/BBS)
