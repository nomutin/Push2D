"""Hyparparameters for example situations."""

from pygame import Color
from pymunk import Vec2d

from .types import CircleParameters, SpaceParameters


class RedAndGreen:
    """
    Hyperparameters for the `Red and Green` situation.

    Attributes
    ----------
    SPACE : SpaceParameters
    AGENT : CircleParameters
    RED : CircleParameters
    GREEN : CircleParameters
    """

    SPACE = SpaceParameters(
        width=300,
        height=225,
        fps=60,
        color=Color("black"),
    )
    AGENT = CircleParameters(
        radius=20,
        position=Vec2d(150, 110),
        color=Color("blue"),
        velocity=110,
    )
    RED = CircleParameters(
        radius=30,
        position=Vec2d(50, 50),
        color=Color("red"),
        velocity=0,
    )
    GREEN = CircleParameters(
        radius=30,
        position=Vec2d(250, 175),
        color=Color("green"),
        velocity=0,
    )
