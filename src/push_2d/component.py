"""Simple physics simulator for the pushing object task."""

from __future__ import annotations

import dataclasses

import pygame
import pymunk
import pymunk.pygame_util
from pygame import Color
from pymunk import Vec2d


@dataclasses.dataclass
class Circle:
    """
    A class to represent a circle object with physics properties.

    Methods
    -------
    __post_init__()
        This method is called after the object has been initialized
        and creates a pymunk physics body and shape for the circle with a
        given position, radius, and color.
    """

    radius: int
    position: Vec2d
    color: list[int]
    velocity: int = 0

    def __post_init__(self) -> None:
        """Create Body and Shape for the circle and set its properties."""
        self._setup_body()
        self._setup_shape()

    def _setup_body(self) -> None:
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = self.position

    def _setup_shape(self) -> None:
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.mass = 1.0
        self.shape.color = self.color
        self.shape.friction = 0.7
        self.shape.elasticity = 0

    def add(self, to: Space) -> Space:
        """
        Add a `Circle` to the simulation space.

        And Connect it to a pivot and gear joint for more realistic physics.

        Parameters
        ----------
        to : Space
            The space to add the circle to.
        """
        to.add(self.body, self.shape)

        static_body = to.static_body

        pivot = pymunk.PivotJoint(static_body, self.body, (0, 0), (0, 0))
        to.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = 1000  # emulate linear friction

        gear = pymunk.GearJoint(static_body, self.body, 0.0, 1.0)
        to.add(gear)
        gear.max_bias = 0  # disable joint correction
        gear.max_force = 5000  # emulate angular friction
        return to


@dataclasses.dataclass
class Agent:
    """A class to represent the agent with physics properties."""

    radius: int
    position: Vec2d
    color: Color
    velocity: int = 0

    def __post_init__(self) -> None:
        """Create body/shape/pivot/gear for the agent."""
        self.velocity = self.velocity
        self._setup_control_body()
        self._setup_body()
        self._setup_shape()
        self._setup_pivot()
        self._setup_gear()

    def _setup_control_body(self) -> None:
        self.control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.control_body.position = self.position

    def _setup_body(self) -> None:
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = self.position

    def _setup_shape(self) -> None:
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.color = self.color
        self.shape.mass = 10
        self.shape.friction = 0.7
        self.shape.elasticity = 0

    def _setup_pivot(self) -> None:
        self.pivot = pymunk.PivotJoint(
            self.control_body,
            self.body,
            (0, 0),
            (0, 0),
        )
        self.pivot.max_bias = 0  # disable joint correction
        self.pivot.max_force = 10000  # emulate linear friction

    def _setup_gear(self) -> None:
        self.gear = pymunk.GearJoint(self.control_body, self.body, 0.0, 1.0)
        self.gear.error_bias = 0  # attempt to fully correct the joint
        self.gear.max_bias = 1.2  # but limit it's angular correction rate
        self.gear.max_force = 50000  # emulate angular friction

    def add(self, to: Space) -> Space:
        """Add an agent to the simulation space."""
        to.add(
            self.control_body,
            self.body,
            self.shape,
            self.pivot,
            self.gear,
        )
        return to


@dataclasses.dataclass
class Wall:
    """A class to represent a wall with physics properties."""

    start_x: int
    start_y: int
    end_x: int
    end_y: int
    radius: int
    color: Color

    def add(self, to: Space) -> Space:
        """Add a wall to the simulation space."""
        wall = pymunk.Segment(
            body=to.static_body,
            a=(self.start_x, self.start_y),
            b=(self.end_x, self.end_y),
            radius=self.radius,
        )
        wall.elasticity = 1.0
        wall.color = pygame.Color(self.color)
        to.add(wall)
        return to


class Space(pymunk.Space):
    """
    A class that represents a 2D space simulation.

    Methods
    -------
    add_circle(circle: Circle) -> None
        Add a circular object to the simulation space along with its
        supporting pivot and gear constraints.
    clear() -> None
        Remove all objects from the simulation space.
    render() -> None
        Apply one step of the physics simulation, clearing the screen
        and rendering the updated state of the simulation.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        color: list[int],
    ) -> None:
        """
        Initialize Space.

        - Pygame screen
        - Physics space
        - Rendering options
        """
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.color = color
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

    def clear(self) -> None:
        """Remove all objects from the simulation space."""
        self.remove(*self.bodies)
        self.remove(*self.constraints)
        self.remove(*self.shapes)

    def render(self) -> None:
        """
        Apply one envornment step.

        includes:
            - One step of the physics simulation
            - Clear the screen
            - Render the updated state of the simulation
        """
        self.screen.fill(self.color)
        self.debug_draw(self.draw_options)
        pygame.display.flip()
        self.step(1 / self.fps)
        self.clock.tick(self.fps)
