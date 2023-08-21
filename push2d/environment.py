"""Push2D Environment API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pygame
from gymnasium import Env
from gymnasium.utils import seeding
from pymunk import Vec2d
from typeguard import typechecked

from .component import Circle, Space
from .types import Act, Obs

if TYPE_CHECKING:
    from .types import CircleParameters, SpaceParameters


class Push2D(Env[Act, Obs]):
    """
    The main Gymnasium class for implementing Play Data environments.

    The main API methods that users of this class need to know are:

    - :meth:`step`
        - Updates an environment with actions returning the next observation.
    - :meth:`reset`
        - Resets the environment to an initial state.
          Returns the first agent observation for an episode and information.
    - :meth:`render`
        - Renders the environments to help visualize what the agent see.
    - :meth:`close`
        - Closes the environment.
    """

    def __init__(
        self,
        space_params: SpaceParameters,
        agent_params: CircleParameters,
        obstacles_params: list[CircleParameters],
    ) -> None:
        """Initialize the environment."""
        super().__init__()
        self.space_params = space_params
        self.agent_params = agent_params
        self.obstacles_params = obstacles_params
        self.space = Space(params=self.space_params)
        self.reset(seed=42)

    @typechecked
    def step(
        self, action: Act,
    ) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        """
        Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        `reset()` to reset this environment's state.

        Parameters
        ----------
        action : np.ndarray
            an action provided by the agent

        Returns
        -------
        observation : np.ndarray
            agent's observation of the current environment
        reward : float
            amount of reward returned after previous action
        terminated : bool
            Whether the agent reaches the terminal state.
        truncated : bool
            Whether the truncation condition outside the scope of
            the MDP is satisfied.
        info : dict
            contains auxiliary diagnostic information
            (helpful for debugging, logging, and sometimes learning)
        """
        vector = Vec2d(*action) - self.agent.body.position
        direction = Vec2d.normalized(vector)
        x, y = round(direction[0], 0), round(direction[1], 0)
        self.agent.body.velocity = Vec2d(x, y) * self.agent.params.velocity
        self.render()
        observation = self._get_observation()
        terminated, truncated, reward = False, False, 1
        info = self._get_object_info()
        return observation, reward, terminated, truncated, info

    @typechecked
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Obs, dict[str, Any]]:
        """
        Reset the environment and return an initial observation.

        This method should also reset the environment's random number
        generator(s) if `seed` is an integer or if the environment has not
        yet initialized a random number generator.

        Parameters
        ----------
        seed : int | None, optional
            The seed that is used to initialize the environment's PRNG.
        options : dict[str, Any] | None, optional
            Additional information to specify how the environment is reset.

        Returns
        -------
        observation : Obs
            Observation of the initial state.
        info : dict
            This dictionary contains auxiliary information
            complementing ``observation``. It should be analogous to
            the ``info`` returned by :meth:`step`.
        """
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        if options is not None:
            self.agent_params = options.get("agent_params", self.agent_params)
            self.obstacles_params = options.get(
                "obstacles_params",
                self.obstacles_params,
            )

        self.space.clear()
        self.agent = Circle(params=self.agent_params)
        self.space.add(self.agent)

        self.obstacles = []
        for obstacle_params in self.obstacles_params:
            obstacle = Circle(params=obstacle_params)
            self.obstacles.append(obstacle)
            self.space.add(obstacle)
        self.space.add_segments()
        self.render()
        observation = self._get_observation()
        info = self._get_object_info()
        return observation, info

    def render(self, _: str = "") -> None:
        """Render the environment."""
        self.space.step()

    def close(self) -> None:
        """Close rendering `pygame` windows."""
        pygame.quit()

    def _get_observation(self) -> Obs:
        """
        Get current observation.

        Returns
        -------
        np.ndarray
            An array containing the current observation state.
        """
        surface = pygame.surfarray.array3d(self.space.screen)
        return np.transpose(surface, (1, 0, 2))

    def _get_object_info(self) -> dict[str, Any]:
        """
        Get current object information.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the current object information.
        """
        return {
            "agent_position": self.agent.body.position,
            "agent_velocity": self.agent.body.velocity,
            "obstacles_position": [p.body.positions for p in self.obstacles],
            "obstacles_velocity": [p.body.velocity for p in self.obstacles],
        }
