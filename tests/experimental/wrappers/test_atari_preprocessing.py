"""Test suite for AtariPreprocessingV0."""
from copy import deepcopy

import numpy as np
import pytest

import gymnasium as gym
from gymnasium.experimental.wrappers import AtariPreprocessingV0
from gymnasium.utils import seeding


"""Tests

- [ ] Noop reset
- [x] Frame skipping
- [ ] Max-pooling
- [ ] Terminate on life loss
- [x] Resize
- [ ] Grayscale observation
- [ ] Scale observation

* Test errors
"""

ATARI_ENV_NAME = "PongNoFrameskip-v4"


# @pytest.mark.parametrize("env_name", [env_name for env_name in gym.envs.registry if "Pong" in env_name])
# def test_environment_versions(env_name):
#     env = gym.make(env_name)
#     AtariPreprocessingV0(env)


def test_atari_reset_randomness():
    print()
    env = gym.make(ATARI_ENV_NAME)

    env.reset(seed=123)
    print(env.np_random.bit_generator.state)
    env.step(env.action_space.sample())
    print(env.np_random.bit_generator.state)

    env = AtariPreprocessingV0(env)
    env.reset(seed=123)
    print(env.np_random.bit_generator.state)
    env.step(env.action_space.sample())
    print(env.np_random.bit_generator.state)

    print(seeding.np_random(seed=123)[0].bit_generator.state)


def test_noop_reset():
    env = AtariPreprocessingV0(gym.make(ATARI_ENV_NAME), noop_max=0)

    rng = deepcopy(env.np_random)


# def test_atari_noop_resets():
#     """Tests atari noop resets"""
#     default_env = AtariPreprocessingV0(gym.make(ATARI_ENV_NAME))
#     default_obs = default_env.reset(seed=123)
#
#     # Check that with zero no-ops is equal to default environment
#     noop_0_env = AtariPreprocessingV0(gym.make(ATARI_ENV_NAME), noop_max=30)  # rng-seed=43 -> no-ops=0
#     noop_0_obs = noop_0_env.reset(seed=123)
#     assert np.all(default_obs[..., -1] == noop_0_obs[..., -1])
#
#     # Check that with 5 no-ops is equal to 5 no-op steps in the default environments
#     noop_5_env = AtariEnv('Breakout', noop_max=30, rng_seed=20)  # rng-seed=20 -> no-ops=5
#     noop_5_obs = noop_5_env.reset(seed=123)
#     [default_env.step(0) for _ in range(5)]
#     assert np.all(default_env.obs[..., -1] == noop_5_obs[..., -1])
#
#     # Check that with 30 no-ops is equal to 30 no-ops steps in the default environment
#     noop_30_env = AtariEnv('Breakout', noop_max=30, rng_seed=12)  # rng-seed=12 -> no-ops=30
#     noop_30_obs = noop_30_env.reset(seed=123)
#     [default_env.step(0) for _ in range(30 - 5)]
#     assert np.all(default_env.obs[..., -1] == noop_30_obs[..., -1])


@pytest.mark.parametrize("screen_size", (84, 80, 105, 160))
def test_screen_size(screen_size):
    """Tests that all the environment observations are as expected"""
    env = gym.make(ATARI_ENV_NAME)
    env = AtariPreprocessingV0(env, screen_size=screen_size)

    obs, _ = env.reset()
    assert obs.shape == (screen_size, screen_size)
    obs, _, _, _, _ = env.step(env.action_space.sample())
    assert obs.shape == (screen_size, screen_size)


def test_frame_skip():
    """Tests that the frame skip, not possible to compare ale-py frameskip parameter as max frame is not implemented"""
    env_1 = AtariPreprocessingV0(gym.make(ATARI_ENV_NAME), frame_skip=1)
    env_2 = AtariPreprocessingV0(gym.make(ATARI_ENV_NAME), frame_skip=2)
    env_4 = AtariPreprocessingV0(gym.make(ATARI_ENV_NAME))  # Frame_skip=4 is default
    env_6 = AtariPreprocessingV0(gym.make(ATARI_ENV_NAME), frame_skip=6)

    # Set all environments to have the same seed
    env_1.action_space.seed(123)

    # Check that the initial frame of all environments are equal
    obs_1, _ = env_1.reset(seed=123)
    obs_2, _ = env_2.reset(seed=123)
    obs_4, _ = env_4.reset(seed=123)
    obs_6, _ = env_6.reset(seed=123)

    assert np.all(obs_1 == obs_2)
    assert np.all(obs_1 == obs_4)
    assert np.all(obs_1 == obs_6)

    action = env_1.action_space.sample()

    def is_zero(obs: np.ndarray):
        return np.all(obs == np.zeros_like(obs, dtype=np.uint8))

    env_1.step(action)
    obs_1, _, _, _, _ = env_1.step(action)
    obs_2, _, _, _, _ = env_2.step(action)
    # Obs 1 = [zeros, reset, frame 1, frame 2]
    # Obs 2 = [zeros, zeros, reset,   frame 2]
    assert is_zero(obs_1[..., :1]) and is_zero(obs_2[..., :2])  # Assert zeros
    assert np.all(obs_1[..., 1] == obs_2[..., 2])  # Assert reset frame
    assert np.all(obs_1[..., 3] == obs_2[..., 3])  # Assert frame 2

    env_1.step(action)
    obs_1, _, _, _, _ = env_1.step(action)
    obs_2, _, _, _, _ = env_2.step(action)
    obs_4, _, _, _, _ = env_4.step(action)
    # Obs 1 = [frame 1, frame 2, frame 3, frame 4]
    # Obs 2 = [zeros,   reset,   frame 2, frame 4]
    # Obs 4 = [zeros,   zeros,   reset,   frame 4]
    assert is_zero(obs_2[..., :1]) and is_zero(obs_4[..., :2])  # Assert zeros
    assert np.all(obs_1[..., 1] == obs_2[..., 2])  # Assert frame 2
    assert np.all(obs_2[..., 1] == obs_4[..., 2])  # Assert reset frame
    assert np.all(obs_1[..., 3] == obs_2[..., 3]) and np.all(
        obs_2[..., 3] == obs_4[..., 3]
    )  # Assert frame 4

    env_1.step(action)
    obs_1, _, _, _, _ = env_1.step(action)
    obs_2, _, _, _, _ = env_2.step(action)
    obs_6, _, _, _, _ = env_6.step(action)
    # Obs 1 = [frame 3, frame 4, frame 5, frame 6]
    # Obs 2 = [reset,   frame 2, frame 4, frame 6]
    # Obs 6 = [zeros,   zeros,   reset,   frame 6]
    assert is_zero(obs_6[..., :2])  # Assert zeros
    assert np.all(obs_1[..., 1] == obs_2[..., 2])  # Assert frame 4
    assert np.all(obs_1[..., 3] == obs_2[..., 3]) and np.all(
        obs_2[..., 3] == obs_6[..., 3]
    )  # Assert frame 6

    env_4.step(action)
    obs_4, _, _, _, _ = env_4.step(action)
    obs_6, _, _, _, _ = env_6.step(action)
    # Obs 4 = [reset, frame 4, frame 8, frame 12]
    # Obs 6 = [zeros, reset,   frame 6, frame 12]
    assert is_zero(obs_6[..., :1])  # Assert zeros
    assert np.all(obs_4[..., 3] == obs_6[..., 3])  # Assert frame 12


def test_atari_termination_on_life_loss():
    """Tests the atari termination on life loss = True and = False"""
    env = AtariPreprocessingV0(gym.make(ATARI_ENV_NAME), terminal_on_life_loss=True)
    env.action_space.seed(123)

    env.reset(seed=123)
    starting_lives = env.lives
    while env.lives == starting_lives:
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            assert env.lives < starting_lives
        else:
            assert env.lives == starting_lives == info["lives"]
