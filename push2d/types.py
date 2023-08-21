"""Type stubs for Push2D."""

from dataclasses import dataclass

from nptyping import Float, Int, NDArray, Shape
from pygame import Color
from pymunk import Vec2d


@dataclass
class SpaceParameters:
    """Parameters for `component.Space`."""

    width: int
    height: int
    fps: int
    color: Color


@dataclass
class CircleParameters:
    """Parameters for `component.Circle`."""

    radius: int
    position: Vec2d
    color: Color
    velocity: int = 0


Act = NDArray[Shape["2"], Float]
Obs = NDArray[Shape["Width, Height, 3"], Int]
