import datetime
import glob
import os
from typing import List

import numpy as np
import pygame
from omegaconf import DictConfig

from components import Circle, Space, mouse_track, rollout
from utils import Keys


class Env:
    def __init__(self, cfg: DictConfig) -> None:
        self.default_caption = "[r]:Reset [s]:Save [q]:Quit "
        pygame.display.set_caption(self.default_caption)
        pygame.key.set_repeat(1000 // cfg.save.fps, 1000 // cfg.save.fps)
        self.cfg = cfg
        self.space = Space.from_dictconfig(cfg.screen)

        self.actions: List[np.ndarray] = []
        self.observations: List[np.ndarray] = []
        self.reset()

    def reset(self) -> None:
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
        surface = pygame.surfarray.array3d(self.space.screen)
        return surface.transpose(1, 0, 2)

    def get_action(self) -> np.ndarray:
        x = int(self.tracker.body.position[0])
        y = int(self.tracker.body.position[1])
        return np.array([x, y])

    def save(self) -> None:
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
        assert action.shape == (2,)
        span = self.cfg.screen.fps // self.cfg.save.fps
        for _ in range(span):
            rollout(
                tracker=self.tracker,
                action=action,
                velocity=self.cfg.tracker.velocity,
            )
            self.space.step()
