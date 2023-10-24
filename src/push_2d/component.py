"""Simple physics simulator for the pushing object task."""

from __future__ import annotations

import pygame
import pymunk
import pymunk.pygame_util
from pygame import Color
from pymunk import Vec2d


class Circle(pymunk.Circle):
    """A class to represent a circle object with physics properties."""

    def __init__(
        self,
        radius: int,
        position: Vec2d,
        color: Color,
    ) -> None:
        """Create Body and Shape for the circle and set its properties."""
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = position
        super().__init__(body=body, radius=radius)
        self.mass = 1.0
        self.color = color
        self.friction = 0.7
        self.elasticity = 0

    def add(self, to: Space) -> Space:
        """
        Add a `Circle` to the simulation space.

        And Connect it to a pivot and gear joint for more realistic physics.

        Parameters
        ----------
        to : Space
            The space to add the circle to.
        """
        to.add(self.body, self)

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


class Agent(pymunk.Circle):
    """A class to represent the agent with physics properties."""

    def __init__(
        self,
        radius: int,
        position: Vec2d,
        color: Color,
        velocity: int = 0,
    ) -> None:
        """Create Body and Shape for the agent and set its properties."""
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = position
        super().__init__(body, radius)
        self.color = color
        self.position = position
        self.velocity = velocity
        self.mass = 10
        self.friction = 0.7
        self.elasticity = 0
        self._setup_control_body()
        self._setup_pivot()
        self._setup_gear()

    def _setup_control_body(self) -> None:
        self.control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.control_body.position = self.position

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
        to.add(self.control_body, self.body, self, self.pivot, self.gear)
        return to


class Segment(pymunk.Segment):
    """A class to represent a wall with physics properties."""

    def __init__(
        self,
        start_position: Vec2d,
        end_position: Vec2d,
        radius: int,
        color: Color,
    ) -> None:
        """Create Body and Shape for the wall and set its properties."""
        super().__init__(
            body=pymunk.Body(body_type=pymunk.Body.STATIC),
            a=start_position,
            b=end_position,
            radius=radius,
        )
        self.color = color
        self.elasticity = 1.0

    def add(self, to: Space) -> Space:
        """Add a wall to the simulation space."""
        to.add(self.body, self)
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
        color: Color,
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
