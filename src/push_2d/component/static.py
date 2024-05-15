# ruff: noqa: PLR0913
"""Define static components."""

import pygame
import pymunk

from .meta import ResettableComponentMeta


class Segment(pymunk.Segment, ResettableComponentMeta):
    """A class to represent a wall with physics properties."""

    def __init__(
        self,
        *,
        x_start_position: int,
        y_start_position: int,
        x_end_position: int,
        y_end_position: int,
        radius: int,
        color: str,
    ) -> None:
        """Create Body and Shape for the wall and set its properties."""
        ResettableComponentMeta.__init__(
            self,
            x_start_position,
            y_start_position,
            x_end_position,
            y_end_position,
            radius,
            color,
        )
        super().__init__(
            body=pymunk.Body(body_type=pymunk.Body.STATIC),
            a=pymunk.Vec2d(x_start_position, y_start_position),
            b=pymunk.Vec2d(x_end_position, y_end_position),
            radius=radius,
        )
        self.color = pygame.Color(color)
        self.elasticity = 1.0

    def add(self, to: pymunk.Space) -> pymunk.Space:
        """Add a wall to the simulation space."""
        to.add(self.body, self)
        return to
