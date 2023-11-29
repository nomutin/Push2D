"""Wrapper for Push2D environment."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pygame
from gymnasium import Wrapper

if TYPE_CHECKING:
    from .environment import Push2D
    from .utils.types import Act, Obs


class ArrowKeyAgentOperator(Wrapper):
    """Wrapper to move the agent by arrow keys."""

    def __init__(self, env: Push2D) -> None:
        """Initialize Wrapper."""
        super().__init__(env=env)
        fps = env.space.fps
        pygame.key.set_repeat(1000 // fps, 1000 // fps)

    @property
    def window_caption(self) -> str:
        """Return the caption of the window."""
        return pygame.display.get_caption()[0]

    @window_caption.setter
    def window_caption(self, caption: str) -> None:
        """Set the caption of the window."""
        pygame.display.set_caption(caption)

    def listen(self) -> Act:
        """Execute `env.step()` by arrow key input."""
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
                self.env.close()  # type: ignore
            if key == pygame.K_r:
                self.reset()
            if key == pygame.K_s:
                self.is_save = True
            if key in direction and event.type == pygame.KEYDOWN:
                action = direction[key]

        return action


class Saver(ArrowKeyAgentOperator):
    def __init__(self, env: Push2D, seq_len: float) -> None:
        super().__init__(env=env)
        self.action_list: list[Act] = []
        self.observation_list: list[Obs] = []
        self.seq_len = seq_len

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Obs, dict[str, Any]]:
        outputs = super().reset(seed=seed, options=options)
        self.action_list.clear()
        self.observation_list.clear()
        self.window_caption = f"{len(self.action_list)}/{self.seq_len}"
        return outputs

    def listen(self) -> Act:
        action = super().listen()
        if not np.all(action == 0):
            observation, *_ = self.step(action)
            self.action_list.append(action)
            self.observation_list.append(observation)
            self.window_caption = f"{len(self.action_list)}/{self.seq_len}"

        if len(self.action_list) == self.seq_len:
            self.save()
        return action

    def save(self) -> None:
        save_directory = Path("data")
        save_directory.mkdir(exist_ok=True)
        actions = np.stack(self.action_list, axis=0)
        observations = np.stack(self.observation_list, axis=0)
        idx = len(list(save_directory.glob("*.npy"))) // 2
        np.save(save_directory / f"action_{idx}.npy", actions)
        np.save(save_directory / f"observation_{idx}.npy", observations)
        self.reset()
