"""Define dynamic components."""

import pygame
import pymunk

from .meta import ResettableComponentMeta


class Circle(pymunk.Circle, ResettableComponentMeta):
    """A class to represent a circle object with physics properties."""

    def __init__(
        self,
        radius: int,
        x_position: int,
        y_position: int,
        color: str,
    ) -> None:
        """Create Body and Shape for the circle and set its properties."""
        ResettableComponentMeta.__init__(
            self,
            radius,
            x_position,
            y_position,
            color,
        )
        body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        body.position = pymunk.Vec2d(x_position, y_position)
        super().__init__(body=body, radius=radius)
        self.mass = 1.0
        self.color = pygame.Color(color)
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


class DynamicBox(ResettableComponentMeta):
    """Movable-opened box."""

    def __init__(
        self,
        radius: int,
        x_position: int,
        y_position: int,
        color: str,
    ) -> None:
        """Initialize a box with 3 segments."""
        ResettableComponentMeta.__init__(
            self,
            radius,
            x_position,
            y_position,
            color,
        )
        self.body = pymunk.Body(body_type=pymunk.Body.DYNAMIC)
        self.body.position = pymunk.Vec2d(x_position, y_position)

        st_points = [(-radius, -radius), (-radius, -radius), (radius, -radius)]
        end_points = [(radius, -radius), (-radius, radius), (radius, radius)]
        self.segments = []
        for s, e in zip(st_points, end_points):
            segment = pymunk.Segment(self.body, s, e, radius=4)
            segment.color = pygame.Color(color)
            segment.elasticity = 0
            segment.friction = 0.7
            segment.mass = 1.0
            self.segments.append(segment)

    def add(self, to: pymunk.Space) -> pymunk.Space:
        """Add a `DynamicBox` to the simulation space."""
        to.add(self.body, *self.segments)

        static_body = to.static_body

        pivot = pymunk.PivotJoint(static_body, self.body, (0, 0), (0, 0))
        to.add(pivot)
        pivot.max_bias = 0  # disable joint correction
        pivot.max_force = 10000  # emulate linear friction

        gear = pymunk.GearJoint(static_body, self.body, 0.0, 1.0)
        to.add(gear)
        gear.max_bias = 0  # disable joint correction
        gear.max_force = 50000  # emulate angular friction
        return to
