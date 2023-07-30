import dataclasses
import datetime
import glob
import os
from random import randint
from typing import List, Tuple

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from omegaconf import DictConfig
from pymunk import Vec2d


class Simulator:
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
        Initializes the simulator class.

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
                key = getattr(event, "key", None)
                if key == pygame.K_q:
                    return
                if key == pygame.K_r:
                    self.reset()
                if key == pygame.K_s:
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


@dataclasses.dataclass
class Circle:
    """
    A class to represent a circle object with physics properties.

    Attributes
    ----------
    radius : int
        The radius of the circle.
    position : Tuple[int,int]
        The (x,y) coordinates of the center of the circle.
    color : str
        The colo name of the circle in pygame.colordict.

    Methods
    -------
    __post_init__()
        This method is called after the object has been initialized
        and creates a pymunk physics body and shape for the circle with a
        given position, radius, and color.
    """

    radius: int
    position: Tuple[int, int]
    color: str

    def __post_init__(self) -> None:
        """
        Create a pymunk Body and Shape for the circle and set its properties.
        """
        self.body = pymunk.Body()
        self.body.position = self.position
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.mass = 1.0
        self.shape.color = pygame.Color(self.color)
        self.shape.elasticity = 1.0

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> "Circle":
        """
        Create a Circle object from a dictionary-like configuration object

        Parameters
        ----------
        cfg : DictConfig
            A DictConfig object containing the configuration of the circle.

        Returns
        -------
        Circle
            A Circle object with properties specified in the `cfg`.
        """
        if "position" in cfg:
            x, y = cfg.position
        elif "screen_width" in cfg and "screen_height" in cfg:
            x = randint(cfg.radius, cfg.screen_width - cfg.radius)
            y = randint(cfg.radius, cfg.screen_height - cfg.radius)
        else:
            raise NotImplementedError

        return Circle(
            radius=cfg.radius,
            position=(x, y),
            color=cfg.color,
        )


def mouse_track(
    tracker: Circle, velocity: float, x_limit: int, y_limit: int
) -> None:
    """
    Using the physics engine pymunk, the Circle object (tracker) is
    given a velocity vector to move it toward the mouse position.

    Parameters
    ----------
    tracker : Circle
        The Circle object to be moved.
    velocity : float
        The velocity at which the tracker should move.
    x_limit : int
        The maximum x-coordinate to which the tracker can move.
    y_limit : int
        The maximum y-coordinate to which the tracker can move.
    """
    mouse_position = pygame.mouse.get_pos()
    mouse_x = np.clip(mouse_position[0], 10, x_limit - 10)
    mouse_y = np.clip(mouse_position[1], 10, y_limit - 10)
    vector = Vec2d(mouse_x, mouse_y) - tracker.body.position
    normalized = Vec2d.normalized(vector)
    x, y = round(normalized[0], 0), round(normalized[1], 0)
    tracker.body.velocity = Vec2d(x, y) * velocity


def rollout(tracker: Circle, action: np.ndarray, velocity: float) -> None:
    vector = Vec2d(*action) - tracker.body.position
    direction = Vec2d.normalized(vector)
    x, y = round(direction[0], 0), round(direction[1], 0)
    tracker.body.velocity = Vec2d(x, y) * velocity


@dataclasses.dataclass
class Space:
    """
    A class that represents a 2D space simulation.

    Attributes
    ----------
    width : int
        The width of the simulation window.
    height : int
        The height of the simulation window.
    fps : int
        The flame refresh rate of the simulation window.
    color : str
        The color of the simulation window.

    Methods
    -------
    add(circle: Circle) -> None
        Add a circular object to the simulation space along with its
        supporting pivot and gear constraints.
    clear() -> None
        Remove all objects from the simulation space.
    step() -> None
        Apply one step of the physics simulation, clearing the screen
        and rendering the updated state of the simulation.
    """

    width: int
    height: int
    fps: int
    color: str

    def __post_init__(self) -> None:
        """
        Initialize the Pygame screen, physics space, and rendering options.
        """
        self.space = pymunk.Space()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    def add(self, circle: Circle) -> None:
        """
        Add a circular object to the simulation space and connect it to a pivot
        and gear joint for more realistic physics.

        Parameters
        ----------
        circle : Circle
            The circular object to add to the simulation space.
        """
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

    def clear(self) -> None:
        """
        Remove all objects from the simulation space.
        """
        self.space.remove(*self.space.bodies)
        self.space.remove(*self.space.constraints)
        self.space.remove(*self.space.shapes)

    def step(self) -> None:
        """
        Apply one step of the physics simulation, clear the screen,
        and render the updated state of the simulation.
        """
        self.screen.fill(self.color)
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.space.step(1 / self.fps)
        self.clock.tick(self.fps)

    def add_segments(self) -> None:
        """
        Create a set of static segments around the edge of the simulation
        space to create walls.
        """

        def wall(a: Tuple[int, int], b: Tuple[int, int]) -> None:
            wall = pymunk.Segment(self.space.static_body, a, b, 1)
            wall.elasticity = 1.0
            wall.color = pygame.Color(self.color)
            self.space.add(wall)

        wall(a=(0, 1), b=(self.width, 1))
        wall(a=(1, 0), b=(1, self.height))
        wall(a=(self.width - 1, 0), b=(self.width - 1, self.height))
        wall(a=(0, self.height - 1), b=(self.width, self.height - 1))

    @classmethod
    def from_dictconfig(cls, config: DictConfig) -> "Space":
        """
        Create a new instance of Space from a DictConfig object.

        Parameters
        ----------
        config : DictConfig
            A DictConfig object with keys "width", "height", "fps", and "color"

        Returns
        -------
        Space
            A new instance of Space
        """
        return Space(
            width=config.width,
            height=config.height,
            fps=config.fps,
            color=config.color,
        )
