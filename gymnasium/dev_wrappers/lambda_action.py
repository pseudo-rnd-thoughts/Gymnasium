"""Lambda action wrapper which apply a function to the provided action."""

from typing import Callable

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperActType


class LambdaActionV0(gym.ActionWrapper[ObsType, WrapperActType]):
    """A wrapper that provides a function to modify the action passed to :meth:`step`."""

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[WrapperActType], ActType],
    ):
        """Initialize LambdaAction.

        Args:
            env (Env): The gymnasium environment
            func (Callable): function to apply to action
        """
        super().__init__(env)

        self.func = func

    def action(self, action: WrapperActType) -> ActType:
        """Apply function to action."""
        return self.func(action)
