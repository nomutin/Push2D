"""Define kinematic components."""


import pygame
import pymunk

from .meta import ResettableComponentMeta


class Circle(pymunk.Circle, ResettableComponentMeta):
    """A class to represent a circle object with physics properties."""

    def __init__(
        self,
        radius: int,
        position: pymunk.Vec2d,
        color: pygame.Color,
    ) -> None:
        """Create Body and Shape for the circle and set its properties."""
        ResettableComponentMeta.__init__(self, radius, position, color)
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = position
        super().__init__(body=body, radius=radius)
        self.mass = 1.0
        self.color = color
        self.friction = 0.7
        self.elasticity = 0

    def add(self, to: pymunk.Space) -> pymunk.Space:
        """Add a `Circle` to the simulation space."""
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
