"""Define the agent classes."""

from __future__ import annotations

import pygame
import pymunk

from .meta import ResettableComponentMeta


class Agent(ResettableComponentMeta):
    """A class to represent the agent with physics properties."""

    def __init__(
        self,
        x_position: int,
        y_position: int,
        color: str,
        velocity: int,
        **kwargs: str | int,
    ) -> None:
        """Create Body and Shape for the agent and set its properties."""
        ResettableComponentMeta.__init__(
            self,
            x_position,
            y_position,
            color,
            velocity,
            **kwargs,
        )
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = pymunk.Vec2d(x_position, y_position)
        self.color = pygame.Color(color)
        self.velocity = velocity

        self._setup_control_body()
        self._setup_pivot()
        self._setup_gear()

    @property
    def shape(self) -> pymunk.Shape:
        """Return the shape of the agent."""
        return self._shape

    @shape.setter
    def shape(self, shape: pymunk.Shape) -> None:
        """Set the shape of the agent."""
        shape.mass = 10
        shape.friction = 0.7
        shape.elasticity = 0
        shape.color = self.color
        shape.position = self.body.position
        self._shape = shape

    def _setup_control_body(self) -> None:
        self.control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.control_body.position = self.body.position

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

    def add(self, to: pymunk.Space) -> pymunk.Space:
        """Add an agent to the simulation space."""
        to.add(self.control_body, self.body, self.shape, self.pivot, self.gear)
        return to


class CircleAgent(Agent):
    """Agent with a circle shape."""

    def __init__(  # noqa: PLR0913
        self,
        x_position: int,
        y_position: int,
        color: str,
        velocity: int,
        radius: int,
    ) -> None:
        """Create shape as `pymunk.Circle`."""
        super().__init__(
            x_position,
            y_position,
            color,
            velocity,
            radius=radius,
        )
        self.shape = pymunk.Circle(self.body, radius)
