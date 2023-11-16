"""Define static components."""

import pygame
import pymunk

from .meta import ResettableComponentMeta


class Segment(pymunk.Segment, ResettableComponentMeta):
    """A class to represent a wall with physics properties."""

    def __init__(
        self,
        start_position: pymunk.Vec2d,
        end_position: pymunk.Vec2d,
        radius: int,
        color: pygame.Color,
    ) -> None:
        """Create Body and Shape for the wall and set its properties."""
        ResettableComponentMeta.__init__(
            self,
            start_position,
            end_position,
            radius,
            color,
        )
        super().__init__(
            body=pymunk.Body(body_type=pymunk.Body.STATIC),
            a=start_position,
            b=end_position,
            radius=radius,
        )
        self.color = color
        self.elasticity = 1.0

    def add(self, to: pymunk.Space) -> pymunk.Space:
        """Add a wall to the simulation space."""
        to.add(self.body, self)
        return to
