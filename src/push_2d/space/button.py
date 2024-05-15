"""Define the Space with buttons and lights."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame
import pymunk

from .meta import Space

if TYPE_CHECKING:
    import pymunk.pygame_util


class Button:
    """A class that represents a button parameter."""

    def __init__(self, x: int, y: int, radius: int, color: str) -> None:
        """Initialize the button with given parameters."""
        self.rect = pygame.Rect(x, y, radius, radius)
        self.color = pygame.Color(color)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the button."""
        pygame.draw.rect(surface=screen, color=self.color, rect=self.rect)


def pre_solve(arbiter: pymunk.Arbiter, space: ButtonSpace, _: dict) -> bool:
    """Pre-solve function for button-pushing handling."""
    shape0 = arbiter.shapes[0].__class__.__name__
    segment = arbiter.shapes[0] if shape0 == "Segment" else arbiter.shapes[1]
    if segment.color == pygame.Color("black"):
        return True
    if space.colors and space.colors[-1] == segment.color:
        return True
    space.colors.append(segment.color)
    return False


class ButtonSpace(Space):
    """A class that represents a space with buttons."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        width: int,
        height: int,
        fps: int,
        color: str,
        buttons: list[Button],
        light_size: int,
    ) -> None:
        """Initialize the ButtonSpace with given parameters."""
        super().__init__(width, height, fps, color)
        self.buttons = buttons
        self.light_size = light_size
        self.colors: list[pygame.Color] = []
        self.handler = self.add_collision_handler(0, 0)
        self.handler.pre_solve = pre_solve

    def render_button(self) -> None:
        """Render all the buttons in the space."""
        for button in self.buttons:
            button.draw(screen=self.screen)

    def render_light(self) -> None:
        """Render the light in the space."""
        for i, color in enumerate(self.colors):
            left = (i * self.light_size) // 2
            pygame.draw.rect(
                surface=self.screen,
                color=color,
                rect=pygame.Rect(
                    left,
                    0,
                    self.light_size // 2,
                    self.light_size,
                ),
            )

    def clear(self) -> None:
        """Clear the space."""
        super().clear()
        self.colors.clear()

    def render(self) -> None:
        """Apply one environment step."""
        self.screen.fill(self.color)
        self.render_button()
        self.render_light()
        self.debug_draw(self.draw_options)
        pygame.display.flip()
        self.step(1 / self.fps)
        self.clock.tick(self.fps)

    def accelerated_render(self) -> None:
        """High speed render(for replay)."""
        self.screen.fill(self.color)
        self.render_button()
        self.render_light()
        self.debug_draw(self.draw_options)
        self.step(1 / self.fps)
