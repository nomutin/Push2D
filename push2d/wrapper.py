"""
Save & Replay API.

To use this, the following command must be executed;

```
poetry install --with dev
```
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame
import torchshow as ts
from gymnasium import Env, Wrapper
from pymunk import Vec2d
from typeguard import typechecked

from push2d.environment import Push2D
from push2d.variants import RedAndGreen

if TYPE_CHECKING:
    from .types import Act, Actions, Obs, Observations


class Saver(Wrapper):
    """Data saver for the Push2D environment."""

    def __init__(
        self,
        env: Env,
        fps: int = 10,
        save_length: int = 200,
    ) -> None:
        """Initialize Saver."""
        super().__init__(env=env)
        self.save_length = save_length
        self.default_caption = "[r]:Reset [s]:Save [q]:Quit "
        pygame.display.set_caption(self.default_caption)
        pygame.key.set_repeat(1000 // fps, 1000 // fps)

        self.actions: list[Act] = []
        self.observations: list[Obs] = []
        self.is_save = False

    def get_action(self) -> Act:
        """Get the action from the environment."""
        action = Vec2d(0, 0)
        direction = {
            pygame.K_a: Vec2d(-1, 0),
            pygame.K_d: Vec2d(1, 0),
            pygame.K_w: Vec2d(0, -1),
            pygame.K_z: Vec2d(0, 1),
        }
        for event in pygame.event.get():
            key = getattr(event, "key", 999)
            if key in direction:
                action += direction[key]
        return np.array(action)

    def follow(self) -> None:
        """
        Execute main loop.

        - It listens for player's input and updates the state accordingly.
        """
        while True:
            action = np.array([0, 0, 0, 0])
            direction = {
                pygame.K_UP: np.array([1, 0, 0, 0]),
                pygame.K_DOWN: np.array([0, 1, 0, 0]),
                pygame.K_LEFT: np.array([0, 0, 1, 0]),
                pygame.K_RIGHT: np.array([0, 0, 0, 1]),
            }

            for event in pygame.event.get():
                key = getattr(event, "key", 999)
                if key == pygame.K_q:
                    self.env.close()
                if key == pygame.K_r:
                    self.reset()
                if key == pygame.K_s:
                    self.is_save = True
                if key in direction and event.type == pygame.KEYDOWN:
                    action += direction[key]

            action = np.array(action)
            observation = self.step(action)[0]

            if self.is_save:
                self.actions.append(action)
                self.observations.append(observation)
                caption = f"{len(self.actions)}/{self.save_length} "
                pygame.display.set_caption(caption)
                if len(self.actions) == self.save_length:
                    self.save()
                    self.is_save = False

    def save(self) -> None:
        """Save the observation/action states."""
        dirname = Path("data")
        Path(dirname).mkdir(parents=True, exist_ok=True)
        num = len(glob.glob(f"{dirname}/*_[0-9]*.npy")) // 2
        np.save(f"{dirname}/raw_action_{num}.npy", np.stack(self.actions))
        np.save(f"{dirname}/raw_obs_{num}.npy", np.stack(self.observations))
        self.actions.clear()
        self.observations.clear()
        pygame.display.set_caption(self.default_caption)

    def replay(self, actions: Actions) -> None:
        """
        Play a saved actions.

        Parameters
        ----------
        actions: Act
            The saved action.
        """
        observation_list = [self.reset()[0]]
        for action in actions:
            observation = self.step(action)[0]
            observation_list.append(observation)
        np.save("replay.npy", np.stack(observation_list))


@typechecked
def save_movie(
    observations: Observations,
) -> None:
    """
    Save tensor data as movie.

    References
    ----------
    * [torchshow](https://github.com/xwying/torchshow)
    """
    ts.show_video(
        x=observations,
        display=True,
        tight_layout=True,
        show_axis=True,
        fps=10,
    )


def save_test() -> None:
    """Test for action/observation save."""
    env = Push2D(
        space_params=RedAndGreen.SPACE,
        agent_params=RedAndGreen.AGENT,
        obstacles_params=[RedAndGreen.RED, RedAndGreen.GREEN],
    )
    saver = Saver(env=env, save_length=100)
    saver.follow()


def rollout_test() -> None:
    """Test for action replay."""
    env = Push2D(
        space_params=RedAndGreen.SPACE,
        agent_params=RedAndGreen.AGENT,
        obstacles_params=[RedAndGreen.RED, RedAndGreen.GREEN],
    )
    actions = np.load("data/8_22_3/raw_action_0.npy")
    saver = Saver(env=env, save_length=100)
    saver.replay(actions=actions)


if __name__ == "__main__":
    save_test()
