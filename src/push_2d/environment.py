"""Push2D Environment API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pygame
from gymnasium import Env, spaces
from hydra.utils import instantiate
from pymunk import Vec2d

from .reward import AbstractRewardFactory
from .utils.config import load_config
from .utils.types import Act, Obs

if TYPE_CHECKING:
    from .component.agent import Agent
    from .component.meta import ResettableComponentMeta
    from .space.meta import Space


class Push2D(Env[Obs, Act]):
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
        space: Space,
        agent: Agent,
        components: list[ResettableComponentMeta],
        reward_factory: type[AbstractRewardFactory] = AbstractRewardFactory,
    ) -> None:
        """Initialize the environment."""
        super().__init__()
        self.space = space
        self.agent = agent
        self.components = components
        self.reward_factory = reward_factory()
        self.default_seed = 42
        self._acceleration = False
        self.reset(seed=self.default_seed)

    def render(self, _: str = "") -> None:
        """
        Render the environment.

        This is not explicitly called,
        `step()` or `reset()` will always call it.
        """
        if self.acceleration:
            self.space.accelerated_render()
        else:
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
            Vec2d(*_action) * self.agent.velocity
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
        options: dict[str, Any] | None = None,  # noqa: ARG002
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

        self.space.clear()
        self.agent.reset()
        self.agent.add(to=self.space)
        for component in self.components:
            component.reset()
            component.add(to=self.space)

        self.render()
        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def close(self) -> None:  # noqa: PLR6301
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
        return {"agent": self.agent, "obstacles": self.components}

    @classmethod
    def from_yaml(cls, setting_name: str) -> Push2D:
        """
        Initialize the environment from a YAML file.

        Parameters
        ----------
        setting_name : str
            Name of YAML file.

        Returns
        -------
        Push2D
            An instance of the environment.
        """
        config = load_config(setting_name=setting_name)
        agent = instantiate(config.agent)
        space = instantiate(config.space)
        components = [instantiate(c) for c in config.components]
        return cls(
            agent=agent,
            space=space,
            components=components,
        )

    @property
    def acceleration(self) -> bool:
        """Return the acceleration flag."""
        return self._acceleration

    @acceleration.setter
    def acceleration(self, flag: bool) -> None:
        """Set the acceleration flag."""
        self._acceleration = flag
