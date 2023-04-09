import pygame
from pygame.event import Event


class Keys:
    """
    A class that provides methods for handling Pygame keyboard input.

    Methods
    -------
    is_quit(event: Event) -> bool
        Returns True if the given Pygame Event object is a Quit event.
    is_key(event: Event) -> bool
        Returns True if the given Pygame Event object is a KEYDOWN event.
    is_q(event: Event) -> bool
        Returns True if the given Pygame Event object is a KEYDOWN event
        and the key pressed was 'q'.
    is_r(event: Event) -> bool
        Returns True if the given Pygame Event object is a KEYDOWN event
        and the key pressed was 'r'.
    is_s(event: Event) -> bool
        Returns True if the given Pygame Event object is a KEYDOWN event
        and the key pressed was 's'.
    """

    @classmethod
    def is_quit(cls, event: Event) -> bool:
        return event.type == pygame.QUIT

    @staticmethod
    def is_key(event: Event) -> bool:
        return event.type == pygame.KEYDOWN

    @classmethod
    def is_q(cls, event: Event) -> bool:
        return cls.is_key(event) and event.key == pygame.K_q

    @classmethod
    def is_r(cls, event: Event) -> bool:
        return cls.is_key(event) and event.key == pygame.K_r

    @classmethod
    def is_s(cls, event: Event) -> bool:
        return cls.is_key(event) and event.key == pygame.K_s
