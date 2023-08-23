"""Hyperparameters for example situations."""

from pygame import Color
from pymunk import Vec2d

from .types import CircleParameters, SpaceParameters


class RedAndGreen:
    """
    Hyperparameters for the `Red and Green` situation.

    Attributes
    ----------
    CENTER : Vec2d
    SPACE : SpaceParameters
    AGENT : CircleParameters
    RED : CircleParameters
    GREEN : CircleParameters
    """

    CENTER = Vec2d(150, 110)
    SPACE = SpaceParameters(
        width=300,
        height=225,
        fps=15,
        color=Color("white"),
    )
    AGENT = CircleParameters(
        radius=20,
        position=CENTER,
        color=Color("blue"),
        velocity=100,
    )
    RED = CircleParameters(
        radius=30,
        position=Vec2d(50, 50),
        color=Color("red"),
    )
    GREEN = CircleParameters(
        radius=30,
        position=Vec2d(250, 175),
        color=Color("green"),
    )
