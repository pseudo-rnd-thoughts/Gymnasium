from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
from gymnasium.functional import FuncEnv


class TestEnv(FuncEnv):
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        super().__init__(options)

    def initial(self, rng: Any) -> npt.NDArray[np.float32]:
        return np.array([0, 0], dtype=np.float32)

    def observation(self, state: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return state

    def transition(self, state: npt.NDArray[np.float32], action: int, rng: None) -> npt.NDArray[np.float32]:
        return state + np.array([0, action], dtype=np.float32)

    def reward(self, state: npt.NDArray[np.float32], action: int, next_state: npt.NDArray[np.float32]) -> float:
        return 1.0 if next_state[1] > 0 else 0.0

    def terminal(self, state: npt.NDArray[np.float32]) -> bool:
        return state[1] > 0


def test_api():
    env = TestEnv()
    state = env.initial(None)
    obs = env.observation(state)
    assert state.shape == (2,)
    assert state.dtype == np.float32
    assert obs.shape == (2,)
    assert obs.dtype == np.float32
    assert np.allclose(obs, state)

    actions = [-1, -2, -5, 3, 5, 2]
    for i, action in enumerate(actions):
        next_state = env.transition(state, action, None)
        assert next_state.shape == (2,)
        assert next_state.dtype == np.float32
        assert np.allclose(next_state, state + np.array([0, action]))

        observation = env.observation(next_state)
        assert observation.shape == (2,)
        assert observation.dtype == np.float32
        assert np.allclose(observation, next_state)

        reward = env.reward(state, action, next_state)
        assert reward == (1.0 if next_state[1] > 0 else 0.0)

        terminal = env.terminal(next_state)
        assert terminal == (i == 5)  # terminal state is in the final action

        state = next_state