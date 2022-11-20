"""Lambda observation wrappers which apply a function to the observation."""

from typing import Callable

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType


class LambdaObservationsV0(gym.ObservationWrapper[WrapperObsType, ActType]):
    """Lambda observation wrapper where a function is provided that is applied to the observation."""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[ObsType], WrapperObsType],
    ):
        """Constructor for the lambda observation wrapper.

        Args:
            env: The environment to wrap
            func: A function that takes
        """
        super().__init__(env)

        self.func = func

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Apply function to the observation."""
        return self.func(observation)
