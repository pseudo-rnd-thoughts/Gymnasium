# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains an ant to run in the +x direction."""
from __future__ import annotations

import os.path
from typing import Any

import jax
import jax.numpy as jnp
from brax import base, math
from brax.generalized import pipeline as g_pipeline
from brax.io import mjcf
from brax.positional import pipeline as p_pipeline
from brax.spring import pipeline as s_pipeline

from gymnasium.experimental import FuncEnv
from gymnasium.experimental.functional import (
    ActType,
    ObsType,
    RewardType,
    StateType,
    TerminalType,
)
from gymnasium.experimental.functional_jax_env import (
    FunctionalJaxEnv,
    FunctionalJaxVectorEnv,
)
from gymnasium.utils import EzPickle


brax_pipelines = {
    "generalized": g_pipeline,
    "spring": s_pipeline,
    "positional": p_pipeline,
}


def pipeline_init(pipeline, sys, q: jnp.ndarray, qd: jnp.ndarray, debug) -> base.State:
    """Initializes the pipeline state."""
    return pipeline.init(sys, q, qd, debug)


def pipeline_step(
    pipeline, sys, pipeline_state: Any, action: jnp.ndarray, n_frames, debug
) -> base.State:
    """Takes a physics step using the physics pipeline."""

    def f(state, _):
        return (
            pipeline.step(sys, state, action, debug),
            None,
        )

    return jax.lax.scan(f, pipeline_state, (), n_frames)[0]


class AntFunctional(FuncEnv):
    __pytree_ignore__ = (
        "backend",
        "pipeline",
    )

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        backend="generalized",
        debug=False,
    ):
        super().__init__()

        path = os.path.join(os.path.abspath(__file__), "assets", "ant.xml")
        sys = mjcf.load(path)

        n_frames = 5
        if backend in ["spring", "positional"]:
            sys = sys.replace(dt=0.005)
            n_frames = 10

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jnp.ones_like(sys.actuator.gear)
                )
            )

        self.sys = sys
        self.backend = backend
        self.pipeline = brax_pipelines[backend]
        self.n_frames = n_frames
        self.debug = debug

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def initial(self, rng: Any) -> StateType:
        """Resets the environment to an initial state."""
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        return pipeline_init(self.pipeline, self.sys, q, qd, self.debug)

    def state_info(self, state: StateType) -> dict:
        zero = jnp.zeros(1)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
        }
        return metrics

    def transition(self, state: StateType, action: ActType, rng: Any) -> StateType:
        return pipeline_step(
            self.pipeline, self.sys, state, action, self.n_frames, self.debug
        )

    def step_info(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> dict:
        velocity = (next_state.x.pos[0] - state.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            min_z, max_z = self._healthy_z_range
            is_healthy = jnp.where(next_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
            is_healthy = jnp.where(next_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy)
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        return dict(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=next_state.x.pos[0, 0],
            y_position=next_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(next_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
        )

    def observation(self, state: StateType) -> ObsType:
        """Observe ant body position and velocities."""
        qpos, qvel = state.q, state.qd

        if self._exclude_current_positions_from_observation:
            qpos = state.q[2:]

        return jnp.concatenate([qpos] + [qvel])

    def terminal(self, state: StateType) -> TerminalType:
        min_z, max_z = self._healthy_z_range

        is_healthy = jnp.where(state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
        is_healthy = jnp.where(state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy)

        if self._terminate_when_unhealthy:
            return 1.0 - is_healthy
        else:
            return 0.0

    def reward(
        self, state: StateType, action: ActType, next_state: StateType
    ) -> RewardType:
        velocity = (next_state.x.pos[0] - state.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            min_z, max_z = self._healthy_z_range
            is_healthy = jnp.where(next_state.x.pos[0, 2] < min_z, x=0.0, y=1.0)
            is_healthy = jnp.where(next_state.x.pos[0, 2] > max_z, x=0.0, y=is_healthy)
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))
        contact_cost = 0.0

        return forward_reward + healthy_reward - ctrl_cost - contact_cost

    @property
    def dt(self) -> jnp.ndarray:
        """The timestep used for each env step."""
        return self.sys.dt * self.n_frames


class AntJaxEnv(FunctionalJaxEnv, EzPickle):
    def __int__(self, render_mode: str | None = None, **kwargs: Any):
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)

        env = AntFunctional(**kwargs)
        env.transform(jax.jit)

        FunctionalJaxEnv.__init__(
            self, env, metadata=self.metadata, render_mode=render_mode
        )


class AntJaxVectorEnv(FunctionalJaxVectorEnv, EzPickle):
    def __init__(
        self,
        num_envs: int,
        render_mode: str | None = None,
        max_episode_steps: int = 1000,
        **kwargs: Any,
    ):
        EzPickle.__init__(
            self,
            num_envs=num_envs,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs,
        )
