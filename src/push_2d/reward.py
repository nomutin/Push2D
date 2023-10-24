"""Reward factory API."""

from __future__ import annotations

from typing import Any


class AbstractRewardFactory:
    """Reward factory interface."""

    @classmethod
    def get_reward(
        cls,
        info: dict[str, Any],  # noqa: ARG003
    ) -> float:
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
