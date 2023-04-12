import datetime
import glob
import os
from typing import List

import numpy as np
import pygame
from components import Circle, Space, mouse_track, rollout
from omegaconf import DictConfig
from utils import Keys

__all__ = ["Env"]


class Env:
    """
    The class represents an environment for the pushing object simulation.

    Attributes
    ----------
    default_caption: str
        The default window caption.
    cfg: DictConfig
        The configuration dictionary containing the game parameters.
    space: Space
        The space containing the pymunk field attributes.
    actions: List[np.ndarray]
        A list of actions(positions) taken by the agent.
    observations: List[np.ndarray]
        A list of observations received by the agent.
    tracker: Circle
        The tracker object.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initializes the Env class.

        Parameters
        ----------
        cfg: DictConfig
            The configuration dictionary containing the game parameters.
        """
        self.default_caption = "[r]:Reset [s]:Save [q]:Quit "
        pygame.display.set_caption(self.default_caption)
        pygame.key.set_repeat(1000 // cfg.save.fps, 1000 // cfg.save.fps)
        self.cfg = cfg
        self.space = Space.from_dictconfig(cfg.screen)

        self.actions: List[np.ndarray] = []
        self.observations: List[np.ndarray] = []
        self.reset()

    def reset(self) -> None:
        """
        Resets the environment.
        """
        self.space.clear()

        self.tracker = Circle.from_dictconfig(self.cfg.tracker)
        self.space.add(self.tracker)

        for obj_config in self.cfg.objects:
            circle = Circle.from_dictconfig(obj_config)
            self.space.add(circle)

        self.space.add_segments()

        self.actions.clear()
        self.observations.clear()
        pygame.display.set_caption(self.default_caption)

    def follow(self) -> None:
        """
        The main loop.
        It listens for player's input and updates the state accordingly.
        """
        while True:
            for event in pygame.event.get():
                if Keys.is_quit(event) or Keys.is_q(event):
                    return
                if Keys.is_r(event):
                    self.reset()
                if Keys.is_s(event):
                    self.actions.append(self.get_action())
                    self.observations.append(self.get_observation())
                    caption = f"{len(self.actions)}/{self.cfg.save.length} "
                    pygame.display.set_caption(caption)
                    if len(self.actions) == self.cfg.save.length:
                        self.save()

            mouse_track(
                tracker=self.tracker,
                velocity=self.cfg.tracker.velocity,
                x_limit=self.cfg.screen.width,
                y_limit=self.cfg.screen.height,
            )
            self.space.step()

    def get_observation(self) -> np.ndarray:
        """
        Returns the current observation.

        Returns
        -------
        np.ndarray
            An array containing the current observation state.
        """
        surface = pygame.surfarray.array3d(self.space.screen)
        return np.transpose(surface, (1, 0, 2))

    def get_action(self) -> np.ndarray:
        """
        Returns the current action.

        Returns
        -------
        np.ndarray
            An array containing the current action state(position).
        """
        x, y = map(int, self.tracker.body.position)
        return np.array([x, y])

    def save(self) -> None:
        """
        Saves the observation/action states.
        """
        now = datetime.datetime.now()
        dirname = os.path.join("data", f"{now.month}_{now.day}_{now.hour}")
        os.makedirs(dirname, exist_ok=True)

        num = len(glob.glob(f"{dirname}/*_[0-9]*.npy")) // 2
        action_path = f"{dirname}/raw_action_{num}.npy"
        observation_path = f"{dirname}/raw_observation_{num}.npy"

        np.save(action_path, np.stack(self.actions))
        np.save(observation_path, np.stack(self.observations))

        self.actions.clear()
        self.observations.clear()
        pygame.display.set_caption(self.default_caption)

    def replay(self, action: np.ndarray) -> None:
        """
        Plays a saved action.

        Parameters
        ----------
        action: np.ndarray
            The saved action.
        """
        assert action.shape == (2,)
        span = self.cfg.screen.fps // self.cfg.save.fps
        for _ in range(span):
            rollout(
                tracker=self.tracker,
                action=action,
                velocity=self.cfg.tracker.velocity,
            )
            self.space.step()
