from random import randint

import pygame
from omegaconf import DictConfig

from components import Circle, Space, mouse_track
from utils import Keys


class Env:
    def __init__(self, config: DictConfig) -> None:
        pygame.display.set_caption("[r]:Reset [s]:Save [q]:Quit ")
        self.config = config
        self.space = Space(
            width=config.screen.width,
            height=config.screen.height,
            fps=config.screen.fps,
            color=config.screen.color,
        )
        self.objects = []
        self.reset()

    def reset(self) -> None:
        self.space.clear()

        self.tracker = Circle(
            radius=self.config.tracker.radius,
            color=self.config.tracker.color,
            position=(
                self.config.screen.width // 2,
                self.config.screen.height // 2,
            ),
        )
        self.space.add(self.tracker)

        for obj in self.config.objects:
            x = randint(obj.radius, self.config.screen.width - obj.radius)
            y = randint(obj.radius, self.config.screen.height - obj.radius)
            circle = Circle(
                radius=obj.radius, color=obj.color, position=(x, y)
            )
            self.objects.append(circle)
            self.space.add(circle)

        self.space.add_segments()

    def save(self) -> None:
        while True:
            for event in pygame.event.get():
                if Keys.is_quit(event) or Keys.is_q(event):
                    pygame.quit()
                    exit()
                if Keys.is_r(event):
                    self.reset()

            mouse_track(
                tracker=self.tracker,
                velocity=self.config.tracker.velocity,
                x_limit=self.config.screen.width,
                y_limit=self.config.screen.height,
            )
            self.space.step()

    def test(self) -> None:
        pass
