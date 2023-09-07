"""Reward factory API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pygame import Color

if TYPE_CHECKING:
    from .types import CircleParameters, SpaceParameters


class AbstractRewardFactory:
    """Reward factory interface."""

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """
        Get reward for the current state of the environment.

        Parameters
        ----------
        info : dict[str, Any]
            Dict obtained from the 2nd return of `gymnasium.env.step()`.

        Returns
        -------
        float
            Normalized from 0.0 to 1.0 is recommended.
        """
        return 1.0


class AbstractRedAndGreenRewardFactory(AbstractRewardFactory):
    """
    Abstract reward factory for `variable.RedAndGreen` environment.

    This class contains class methods to identify the circle
    and estimate the location of the circle.
    """

    margin = 30

    @classmethod
    def circle_is_left(
        cls,
        circle_parameters: CircleParameters,
        space_parameters: SpaceParameters,  # noqa: ARG003
    ) -> bool:
        """Get whether the circle is on the left side of the space."""
        x_position = circle_parameters.position[0]
        return x_position < circle_parameters.radius + cls.margin

    @classmethod
    def circle_is_right(
        cls,
        circle_parameters: CircleParameters,
        space_parameters: SpaceParameters,
    ) -> bool:
        """Get whether the circle is on the right side of the space."""
        x_position = circle_parameters.position[0]
        margin = circle_parameters.radius + cls.margin
        return x_position > space_parameters.width - margin

    @classmethod
    def circle_is_top(
        cls,
        circle_parameters: CircleParameters,
        space_parameters: SpaceParameters,  # noqa: ARG003
    ) -> bool:
        """Get whether the circle is on the top side of the space."""
        y_position = circle_parameters.position[1]
        return y_position < circle_parameters.radius + cls.margin

    @classmethod
    def circle_is_bottom(
        cls,
        circle_parameters: CircleParameters,
        space_parameters: SpaceParameters,
    ) -> bool:
        """Get whether the circle is on the bottom side of the space."""
        y_position = circle_parameters.position[1]
        margin = circle_parameters.radius + cls.margin
        return y_position > space_parameters.height - margin

    @classmethod
    def circle_is_top_left(
        cls,
        circle_parameters: CircleParameters,
        space_parameters: SpaceParameters,
    ) -> bool:
        """Get whether the circle is on the top left side of the space."""
        is_top = cls.circle_is_top(circle_parameters, space_parameters)
        is_left = cls.circle_is_left(circle_parameters, space_parameters)
        return is_top and is_left

    @classmethod
    def circle_is_top_right(
        cls,
        circle_parameters: CircleParameters,
        space_parameters: SpaceParameters,
    ) -> bool:
        """Get whether the circle is on the top right of the space."""
        is_top = cls.circle_is_top(circle_parameters, space_parameters)
        is_right = cls.circle_is_right(circle_parameters, space_parameters)
        return is_top and is_right

    @classmethod
    def circle_is_bottom_left(
        cls,
        circle_parameters: CircleParameters,
        space_parameters: SpaceParameters,
    ) -> bool:
        """Get whether the circle is on the bottom left of the space."""
        is_bottom = cls.circle_is_bottom(circle_parameters, space_parameters)
        is_left = cls.circle_is_left(circle_parameters, space_parameters)
        return is_bottom and is_left

    @classmethod
    def circle_is_bottom_right(
        cls,
        circle_parameters: CircleParameters,
        space_parameters: SpaceParameters,
    ) -> bool:
        """Get whether the circle is on the bottom right of the space."""
        is_bottom = cls.circle_is_bottom(circle_parameters, space_parameters)
        is_right = cls.circle_is_right(circle_parameters, space_parameters)
        return is_bottom and is_right

    @staticmethod
    def get_red_circle(info: dict[str, Any]) -> CircleParameters:
        """Identify red circle from object info."""
        for obstacle in info["obstacles"]:
            if obstacle.color == Color("red"):
                return obstacle
        return info["obstacles"][0]

    @staticmethod
    def get_green_circle(info: dict[str, Any]) -> CircleParameters:
        """Identify green circle from object info."""
        for obstacle in info["obstacles"]:
            if obstacle.color == Color("green"):
                return obstacle
        return info["obstacles"][0]


class TopRightRedTopLeftGreen(AbstractRedAndGreenRewardFactory):
    r"""
    Rewarded when red in top left and green in top right.

    ```
    ┌────────────┐
    │R          G│
    │            │
    │            │
    │            │
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_top_left = cls.circle_is_top_left(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_top_right = cls.circle_is_top_right(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_top_left:
            reward += 0.5
        if green_is_top_right:
            reward += 0.5
        return reward


class TopLeftGreenTopRightRed(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when green in top left and red in top right.

    ```
    ┌────────────┐
    │G          R│
    │            │
    │            │
    │            │
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_top_right = cls.circle_is_top_right(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_top_left = cls.circle_is_top_left(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_top_right:
            reward += 0.5
        if green_is_top_left:
            reward += 0.5
        return reward


class TopLeftRedBottomLeftGreen(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when red in top left and green in bottom left.

    ```
    ┌────────────┐
    │R           │
    │            │
    │            │
    │G           │
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_top_left = cls.circle_is_top_left(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_bottom_left = cls.circle_is_bottom_left(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_top_left:
            reward += 0.5
        if green_is_bottom_left:
            reward += 0.5
        return reward


class TopLeftGreenBottomLeftRed(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when green in top left and red in bottom left.

    ```
    ┌────────────┐
    │G           │
    │            │
    │            │
    │R           │
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_bottom_left = cls.circle_is_bottom_left(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_top_left = cls.circle_is_top_left(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_bottom_left:
            reward += 0.5
        if green_is_top_left:
            reward += 0.5
        return reward


class TopLeftRedBottomRightGreen(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when red in top left and green in bottom right.

    ```
    ┌────────────┐
    │R           │
    │            │
    │            │
    │           G│
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_top_left = cls.circle_is_top_left(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_bottom_right = cls.circle_is_bottom_right(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_top_left:
            reward += 0.5
        if green_is_bottom_right:
            reward += 0.5
        return reward


class TopLeftGreenBottomRightRed(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when green in top left and red in bottom right.

    **************
    *G           *
    *            *
    *            *
    *            *
    *           R*
    **************
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_bottom_right = cls.circle_is_bottom_right(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_top_left = cls.circle_is_top_left(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_bottom_right:
            reward += 0.5
        if green_is_top_left:
            reward += 0.5
        return reward


class RightTopRedRightBottomGreen(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when red in right top and green in right bottom.

    ```
    ┌────────────┐
    │           R│
    │            │
    │            │
    │           G│
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_top_right = cls.circle_is_top_right(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_bottom_right = cls.circle_is_bottom_right(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_top_right:
            reward += 0.5
        if green_is_bottom_right:
            reward += 0.5
        return reward


class RightTopGreenRightBottomGreen(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when green in right top and red in right bottom.

    ```
    ┌────────────┐
    │           G│
    │            │
    │            │
    │           R│
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_bottom_right = cls.circle_is_bottom_right(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_top_right = cls.circle_is_top_right(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_bottom_right:
            reward += 0.5
        if green_is_top_right:
            reward += 0.5
        return reward


class TopRightRedBottomLeftGreen(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when red in top right and green in bottom left.

    ```
    ┌────────────┐
    │           R│
    │            │
    │            │
    │G           │
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_top_right = cls.circle_is_top_right(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_bottom_left = cls.circle_is_bottom_left(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_top_right:
            reward += 0.5
        if green_is_bottom_left:
            reward += 0.5
        return reward


class TopRightGreenBottomLeftRed(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when green in top right and red in bottom left.

    ```
    ┌────────────┐
    │           G│
    │            │
    │            │
    │R           │
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)
        red_is_bottom_left = cls.circle_is_bottom_left(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_top_right = cls.circle_is_top_right(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )
        reward = 0.0
        if red_is_bottom_left:
            reward += 0.5
        if green_is_top_right:
            reward += 0.5
        return reward


class BottomLeftRedBottomRightGreen(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when red in bottom left and green in bottom right.

    ```
    ┌────────────┐
    │            │
    │            │
    │            │
    │R          G│
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)

        red_is_bottom_left = cls.circle_is_bottom_left(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_bottom_right = cls.circle_is_bottom_right(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )

        reward = 0.0
        if red_is_bottom_left:
            reward += 0.5
        if green_is_bottom_right:
            reward += 0.5
        return reward


class BottomLeftGreenBottomRightRed(AbstractRedAndGreenRewardFactory):
    """
    Rewarded when green in bottom left and red in bottom right.

    ```
    ┌────────────┐
    │            │
    │            │
    │            │
    │G          R│
    └────────────┘
    ```
    """

    @classmethod
    def get_reward(cls, info: dict[str, Any]) -> float:
        """Earn 0.5 for each condition met."""
        red_circle = cls.get_red_circle(info=info)
        green_circle = cls.get_green_circle(info=info)

        red_is_bottom_right = cls.circle_is_bottom_right(
            circle_parameters=red_circle,
            space_parameters=info["space"],
        )
        green_is_bottom_left = cls.circle_is_bottom_left(
            circle_parameters=green_circle,
            space_parameters=info["space"],
        )

        reward = 0.0
        if red_is_bottom_right:
            reward += 0.5
        if green_is_bottom_left:
            reward += 0.5
        return reward
