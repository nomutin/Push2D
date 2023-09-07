"""Push2D Environment API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pygame
from gymnasium import Env, spaces
from pymunk import Vec2d

from .component import Agent, Circle, Space
from .reward import AbstractRewardFactory
from .types import CircleParameters, SpaceParameters

if TYPE_CHECKING:
    from .types import Act, Obs


class Push2D(Env):
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

    observation_space = spaces.Box(-np.inf, np.inf, shape=(3,))
    action_space = spaces.MultiDiscrete([2, 2, 2, 2])

    def __init__(
        self,
        space_params: SpaceParameters,
        agent_params: CircleParameters,
        obstacles_params: list[CircleParameters],
        reward_factory: type[AbstractRewardFactory] = AbstractRewardFactory,
    ) -> None:
        """Initialize the environment."""
        super().__init__()
        self.space_params = space_params
        self.agent_params = agent_params
        self.obstacles_params = obstacles_params
        self.reward_factory = reward_factory()
        self.space = Space(params=self.space_params)
        self.default_seed = 42
        self.reset(seed=self.default_seed)

    def render(self, _: str = "") -> None:
        """
        Render the environment.

        This is not explicitly called,
        `step()` or `reset()` will always call it.
        """
        self.space.render()

    def step(
        self,
        action: Act,
    ) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        """
        Run one time step of the environment's dynamics.

        When end of episode is reached, you are responsible for calling
        `reset()` to reset this environment's state.

        Parameters
        ----------
        action : Act
            an action provided by the agent

        Returns
        -------
        observation : Obs
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
        directions = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        _action = np.dot(action, directions)
        self.agent.control_body.velocity = (
            Vec2d(*_action) * self.agent_params.velocity
        )
        self.render()
        observation = self._get_observation()
        terminated, truncated = False, False
        info = self._get_info()
        reward = self.reward_factory.get_reward(info)
        return observation, reward, terminated, truncated, info

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
            The seed that is used to initialize the environment's RNG.
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
        self.default_seed = seed if seed is not None else self.default_seed
        np.random.default_rng(self.default_seed)

        if options is not None:
            self.agent_params = options.get("agent_params", self.agent_params)
            self.obstacles_params = options.get(
                "obstacles_params",
                self.obstacles_params,
            )

        self.space.clear()

        self.agent = Agent(params=self.agent_params)
        self.space.add_agent(agent=self.agent)

        self.obstacles = []
        for obstacle_params in self.obstacles_params:
            obstacle = Circle(params=obstacle_params)
            self.obstacles.append(obstacle)
            self.space.add_circle(circle=obstacle)

        self.space.add_segments()
        self.render()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def close(self) -> None:
        """Close rendering `pygame` windows."""
        pygame.quit()

    def _get_observation(self) -> Obs:
        """
        Get current observation.

        Returns
        -------
        Obs
            An array containing the current observation state.
        """
        surface = pygame.surfarray.array3d(self.space.screen)
        return np.transpose(surface, (1, 0, 2))

    def _get_info(self) -> dict[str, Any]:
        """
        Get current object/environment information.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the current object/environment information.
        """
        space_parameters = SpaceParameters(
            width=self.space.width,
            height=self.space.height,
            fps=self.space.fps,
            color=self.space.color,
        )
        agent_parameters = CircleParameters(
            radius=self.agent.params.radius,
            position=self.agent.body.position,
            color=self.agent.params.color,
            velocity=self.agent_params.velocity,
        )
        obstacles_parameters = []
        for obstacle in self.obstacles:
            obstacle_parameters = CircleParameters(
                radius=obstacle.params.radius,
                position=obstacle.body.position,
                color=obstacle.params.color,
                velocity=obstacle.body.velocity,
            )
            obstacles_parameters.append(obstacle_parameters)
        return {
            "space": space_parameters,
            "agent": agent_parameters,
            "obstacles": obstacles_parameters,
        }
