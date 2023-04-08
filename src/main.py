import dataclasses
import random
from typing import Tuple

import pygame
import pymunk
import pymunk.pygame_util
from omegaconf import DictConfig
from pymunk import Vec2d

# define some constants
WIDTH, HEIGHT = 600, 480
FPS = 120
RADIUS_BLUE = 20
RADIUS_RED = 30
n_red_circles = 10
FPS = 60


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


def mouse_track(circle: Circle) -> None:
    mouse_position = pygame.mouse.get_pos()
    velocity = Vec2d(*mouse_position) - circle.body.position
    circle.body.velocity = velocity


class Space:
    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 60,
        background_color: str = "white",
    ) -> None:
        self.space = pymunk.Space()
        self.fps = fps
        self.background_color = background_color
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

        # Create segments around the edge of the screen.
        self._build_wall(a=(0, 1), b=(width, 1))  # top
        self._build_wall(a=(1, 0), b=(1, height))  # left
        self._build_wall(a=(width - 1, 0), b=(width - 1, height))  # right
        self._build_wall(a=(0, height - 1), b=(width, height - 1))  # bottom

    def _build_wall(self, a: Tuple[int, int], b: Tuple[int, int]) -> None:
        wall = pymunk.Segment(self.space.static_body, a, b, 1)
        wall.elasticity = 1.0
        wall.color = pygame.Color(self.background_color)
        self.space.add(wall)

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
        self.screen.fill(self.background_color)
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.space.step(1 / self.fps)
        self.clock.tick(FPS)


class Env:
    def __init__(self, config: DictConfig) -> None:
        pygame.init()
        self.config = config
        self.space = Space(
            width=config.width, height=config.height, fps=config.fps
        )


def main() -> None:
    pygame.init()

    space = Space(WIDTH, HEIGHT, FPS)
    # create the blue circle
    blue_circle = Circle(
        radius=RADIUS_BLUE,
        color="green",
        position=(1, 1),
    )
    space.add(blue_circle)

    # create the red circles

    for i in range(n_red_circles):
        x = random.randint(RADIUS_RED, WIDTH - RADIUS_RED)
        y = random.randint(RADIUS_RED, HEIGHT - RADIUS_RED)
        red_circle = Circle(radius=RADIUS_RED, color="red", position=(x, y))
        space.add(red_circle)

    # main loop
    while True:
        # event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        mouse_track(blue_circle)
        space.step()


if __name__ == "__main__":
    main()
