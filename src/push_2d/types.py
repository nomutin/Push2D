"""Type stubs for Push2D."""

from __future__ import annotations

from typing import Any

from nptyping import Int, NDArray, Shape

# up, down, left, right
Act = NDArray[Shape["4"], Int]
Obs = NDArray[Shape["Width, Height, 3"], Int]


class Resettable:
    """A metaclass that adds a reset functionality to a class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the class and store the initial state."""
        self._initial_state = (args, kwargs)

    def reset(self) -> None:
        """Reset the class to its initial state."""
        args, kwargs = self._initial_state
        self.__init__(*args, **kwargs)  # type: ignore
