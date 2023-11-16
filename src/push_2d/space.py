"""Define extension of `pymunk.Space` classes."""


from __future__ import annotations

import pygame
import pymunk
import pymunk.pygame_util


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
        color: str,
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
        self.color = pygame.Color(color)
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

    def accelerated_render(self) -> None:
        """High speed render(for replay)."""
        self.screen.fill(self.color)
        self.debug_draw(self.draw_options)
        self.step(1 / self.fps)
