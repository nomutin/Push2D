# Push-2D

![python](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![lint](https://github.com/nomutin/push2d-simulator/actions/workflows/lint.yml/badge.svg)](https://github.com/nomutin/push2d-simulator/actions/workflows/lint.yml)

![demo](https://github.com/nomutin/Push-2D/assets/48053582/a0283860-ac3f-4a1c-b4e3-5460570c66f6)

## 🗂️ Usage

### ⚙️ Install

```shell
poetry add git+https://github.com/nomutin/Push-2D.git
```

### 🏋️‍♀️ Gymnasium API

```python
from push_2d.core import Push2D

env = Push2D.from_yaml("")
env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, *_ = env.step(action=action)
```

### ⌨️ Keyboard API

```python
import numpy as np

from push_2d.core import Push2D
from push_2d.wrapper import ArrowKeyAgentOperator

env = Push2D.from_yaml("")
operator = ArrowKeyAgentOperator(env=env)
operator.reset(seed=42)

while True:
    action = operator.listen()
    if not np.all(action == 0):
        env.step(action=action)
```

### 📀 Saver API

```python
from push_2d.core import Push2D
from push_2d.wrapper import Saver

env = Push2D.from_yaml("")
saver = Saver(env=env, fps=RedAndGreen.SPACE.fps, seq_len=300)
saver.reset(seed=42)

while True:
    saver.listen()
```

## 📚 References

- [Pymunk](http://www.pymunk.org/en/latest/)
- [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [BringBackShapes Environment](https://github.com/arnavkj1995/BBS)
