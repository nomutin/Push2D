import dataclasses
from typing import Tuple

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk import Vec2d


@dataclasses.dataclass
class Circle:
    radius: int
    position: Tuple[int, int]
    color: str

    def __post_init__(self) -> None:
        self.body = pymunk.Body()
        self.body.position = self.position
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.mass = 1.0
        self.shape.color = pygame.Color(self.color)


def mouse_track(
    tracker: Circle, velocity: float, x_limit: int, y_limit: int
) -> None:
    mouse_position = pygame.mouse.get_pos()
    mouse_x = np.clip(mouse_position[0], 10, x_limit - 10)
    mouse_y = np.clip(mouse_position[1], 10, y_limit - 10)
    vector = Vec2d(mouse_x, mouse_y) - tracker.body.position
    normalized = Vec2d.normalized(vector)
    x, y = round(normalized[0], 0), round(normalized[1], 0)
    tracker.body.velocity = Vec2d(x, y) * velocity


@dataclasses.dataclass
class Space:
    width: int
    height: int
    fps: int
    color: str

    def __post_init__(self) -> None:
        self.space = pymunk.Space()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    def add(self, circle: Circle) -> None:
        self.space.add(circle.body, circle.shape)

        static_body = self.space.static_body

        pivot = pymunk.PivotJoint(static_body, circle.body, (0, 0), (0, 0))
        self.space.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = 1000  # emulate linear friction

        gear = pymunk.GearJoint(static_body, circle.body, 0.0, 1.0)
        self.space.add(gear)
        gear.max_bias = 0  # disable joint correction
        gear.max_force = 5000  # emulate angular friction

    def step(self) -> None:
        self.screen.fill(self.color)
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.space.step(1 / self.fps)
        self.clock.tick(self.fps)

    def add_segments(self) -> None:
        # Create segments around the edge of the screen.
        def wall(a: Tuple[int, int], b: Tuple[int, int]) -> None:
            wall = pymunk.Segment(self.space.static_body, a, b, 1)
            wall.elasticity = 1.0
            wall.color = pygame.Color(self.color)
            self.space.add(wall)

        wall(a=(0, 1), b=(self.width, 1))
        wall(a=(1, 0), b=(1, self.height))
        wall(a=(self.width - 1, 0), b=(self.width - 1, self.height))
        wall(a=(0, self.height - 1), b=(self.width, self.height - 1))
