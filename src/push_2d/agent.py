"""Define the agent classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymunk

from .component.meta import ResettableComponentMeta

if TYPE_CHECKING:
    import pygame


class Agent(pymunk.Circle, ResettableComponentMeta):
    """A class to represent the agent with physics properties."""

    def __init__(
        self,
        radius: int,
        position: pymunk.Vec2d,
        color: pygame.Color,
        velocity: int = 0,
    ) -> None:
        """Create Body and Shape for the agent and set its properties."""
        ResettableComponentMeta.__init__(
            self,
            radius,
            position,
            color,
            velocity,
        )
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

    def add(self, to: pymunk.Space) -> pymunk.Space:
        """Add an agent to the simulation space."""
        to.add(self.control_body, self.body, self, self.pivot, self.gear)
        return to
