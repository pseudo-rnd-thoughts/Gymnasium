"""Test lambda reward wrapper."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest

import gymnasium as gym
from gymnasium.error import InvalidBound
from gymnasium.wrappers import ClipRewardsV0, LambdaRewardV0

ENV_ID = "CartPole-v1"
DISCRETE_ACTION = 0
NUM_ENVS = 3
SEED = 0


@pytest.mark.parametrize(
    ("reward_fn", "expected_reward"),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward(reward_fn: Callable[[int], int], expected_reward: int):
    """Test lambda reward.

    Tests if function is correctly applied
    to reward.
    """
    env = gym.make(ENV_ID)
    env = LambdaRewardV0(env, reward_fn)
    env.reset(seed=SEED)

    _, rew, _, _, _ = env.step(DISCRETE_ACTION)

    assert rew == expected_reward


@pytest.mark.parametrize(
    (
        "reward_fn",
        "expected_reward",
    ),
    [(lambda r: 2 * r + 1, 3)],
)
def test_lambda_reward_within_vector(
    reward_fn: Callable[[int], int], expected_reward: int
):
    """Test lambda reward in vectorized environment.

    Tests if function is correctly applied
    to reward in a vectorized environment.
    """
    actions = [DISCRETE_ACTION for _ in range(NUM_ENVS)]
    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    env = LambdaRewardV0(env, reward_fn)
    env.reset(seed=SEED)

    _, rew, _, _, _ = env.step(actions)

    assert np.alltrue(rew == expected_reward)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0.0, 0.5, 0.5)],
)
def test_clip_reward(
    lower_bound: float | None, upper_bound: float | None, expected_reward: float
):
    """Test reward clipping.
    Test if reward is correctly clipped
    accordingly to the input args.
    """
    env = gym.make(ENV_ID)
    env = ClipRewardsV0(env, lower_bound, upper_bound)
    env.reset(seed=SEED)
    _, rew, _, _, _ = env.step(DISCRETE_ACTION)

    assert rew == expected_reward


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "expected_reward"),
    [(None, 0.5, 0.5), (0, None, 1), (0, 0.5, 0.5)],
)
def test_clip_reward_within_vector(
    lower_bound: float | None, upper_bound: float | None, expected_reward: float
):
    """Test reward clipping in vectorized environment.
    Test if reward is correctly clipped
    accordingly to the input args in a vectorized environment.
    """
    actions = [DISCRETE_ACTION for _ in range(NUM_ENVS)]

    env = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
    env = ClipRewardsV0(env, lower_bound, upper_bound)
    env.reset(seed=SEED)

    _, rew, _, _, _ = env.step(actions)

    assert np.alltrue(rew == expected_reward)


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound"),
    [(None, None), (1, -1), (np.array([1, 1]), np.array([0, 0]))],
)
def test_clip_reward_incorrect_params(
    lower_bound: int | npt.NDArray[Any] | None,
    upper_bound: int | npt.NDArray[Any] | None,
):
    """Test reward clipping with incorrect params.
    Test whether passing wrong params to clip_rewards
    correctly raise an exception.
    clip_rewards should raise an exception if, both low and upper
    bound of reward are `None` or if upper bound is lower than lower bound.
    """
    env = gym.make(ENV_ID)

    with pytest.raises(InvalidBound):
        env = ClipRewardsV0(env, lower_bound, upper_bound)
