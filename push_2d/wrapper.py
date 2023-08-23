"""Wrapper for Push2D environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pygame
from gymnasium import Wrapper

if TYPE_CHECKING:
    from .core import Push2D
    from .types import Act, Obs


__all__ = ["ArrowKeyAgentOperator"]


class ArrowKeyAgentOperator(Wrapper):
    """Wrapper to move the agent by arrow keys."""

    def __init__(self, env: Push2D, fps: int) -> None:
        """
        Initialize Wrapper.

        Parameters
        ----------
        env : Push2D
            `Push2D` environment instance.
        fps : int
            operating fps.
            It should be equal to `Push2D.space.fps`.
        """
        super().__init__(env=env)
        self.fps = fps
        pygame.key.set_repeat(1000 // fps, 1000 // fps)

    @property
    def window_caption(self) -> str:
        """Return the caption of the window."""
        return pygame.display.get_caption()[0]

    @window_caption.setter
    def window_caption(self, caption: str) -> None:
        """Set the caption of the window."""
        pygame.display.set_caption(caption)

    def listen(self) -> tuple[Obs, Act]:
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
                self.env.close()
            if key == pygame.K_r:
                self.reset()
            if key == pygame.K_s:
                self.is_save = True
            if key in direction and event.type == pygame.KEYDOWN:
                action = direction[key]

        return self.step(action)[0], action
