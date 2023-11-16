"""Define meta classes."""

from typing import Any

import pymunk


class AbstractComponent:
    """An abstract class for components."""

    def add(self, to: pymunk.Space) -> pymunk.Space:
        """Add a component to the simulation space."""
        raise NotImplementedError


class ResettableComponentMeta(AbstractComponent):
    """A metaclass that adds a reset/add to space functionality to a class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the class and store the initial state."""
        self._initial_state = (args, kwargs)

    def reset(self) -> None:
        """Reset the class to its initial state."""
        args, kwargs = self._initial_state
        self.__init__(*args, **kwargs)  # type: ignore
