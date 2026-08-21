r"""Public typing vocabulary shared by Gymnasium's generic classes.

This module documents the *names* used to parameterise Gymnasium's generic
classes and provides the concrete type aliases they rely on (precedent:
:mod:`numpy.typing`).

Since Gymnasium adopted :pep:`695` type-parameter syntax, generic classes
declare their type parameters inline, e.g.::

    class Env[ObsType = Any, ActType = Any]: ...

:pep:`695` type parameters are lexically scoped to the class or function that
declares them, so there is no longer a shared :class:`~typing.TypeVar` object to
import — each class owns its own parameters. The names below are kept as
importable :class:`~typing.TypeVar`\ s purely for backwards compatibility with
downstream code that referenced them (for example to parameterise its own
generic subclasses); new code should use :pep:`695` syntax instead.

Every name declares ``default=Any`` (:pep:`696`), matching the inline defaults on
Gymnasium's own classes, so a class may be subscripted with as few or as many
arguments as desired and any omitted argument falls back to ``Any``.

The single-environment vocabulary parameterises
:class:`gymnasium.Env` ``[ObsType, ActType]`` and
:class:`gymnasium.Wrapper` ``[WrapperObsType, WrapperActType, ObsType, ActType]``;
the vector-environment vocabulary parameterises
:class:`gymnasium.vector.VectorEnv` ``[VectorObsType, VectorActType, VectorRewardType, VectorBoolType]``
and its wrappers. Each name's meaning is documented on the name itself below.
"""

from typing import Any, TypeVar

import numpy as np

__all__ = [
    # Single Agent Env
    "ObsType",
    "ActType",
    "RenderFrame",
    "WrapperObsType",
    "WrapperActType",
    # Vector Env
    "VectorObsType",
    "VectorActType",
    "VectorRewardType",
    "VectorBoolType",
    "VectorWrappedObsType",
    "VectorWrappedActType",
    "VectorWrappedRewardType",
    # Deprecated
    "ArrayType",
]

type RenderFrame = str | np.ndarray | tuple[np.ndarray, np.ndarray]
"""A single frame returned by :meth:`~gymnasium.Env.render` (a concrete alias, not a type parameter)."""

# Single-environment vocabulary
ObsType = TypeVar("ObsType", default=Any)
"""The observation type of an :class:`~gymnasium.Env`, i.e. what :meth:`~gymnasium.Env.reset` and :meth:`~gymnasium.Env.step` return and :attr:`~gymnasium.Env.observation_space` contains."""

ActType = TypeVar("ActType", default=Any)
"""The action type of an :class:`~gymnasium.Env`, i.e. what :meth:`~gymnasium.Env.step` accepts and :attr:`~gymnasium.Env.action_space` contains."""

WrapperObsType = TypeVar("WrapperObsType", default=Any)
"""The observation type a :class:`~gymnasium.Wrapper` exposes to its user, possibly different from the wrapped environment's :data:`ObsType`."""

WrapperActType = TypeVar("WrapperActType", default=Any)
"""The action type a :class:`~gymnasium.Wrapper` accepts from its user, possibly different from the wrapped environment's :data:`ActType`."""

# Vector-environment vocabulary
VectorObsType = TypeVar("VectorObsType", default=Any)
"""The batched observation type of a :class:`~gymnasium.vector.VectorEnv`."""

VectorActType = TypeVar("VectorActType", default=Any)
"""The batched action type of a :class:`~gymnasium.vector.VectorEnv`."""

VectorRewardType = TypeVar("VectorRewardType", default=Any)
"""The batched reward array type of a :class:`~gymnasium.vector.VectorEnv`, typically ``np.ndarray`` of ``float64``."""

VectorBoolType = TypeVar("VectorBoolType", default=Any)
"""The batched termination/truncation array type of a :class:`~gymnasium.vector.VectorEnv`, typically ``np.ndarray`` of ``bool``."""

# `Wrapped` variants are the wrapped (inner) environment's types for wrappers that
# transform observations, actions or rewards. On Gymnasium's own classes these
# default to the wrapper's own type so that a same-type wrapper doesn't need to
# repeat itself; as standalone TypeVars they can only default to `Any`.
VectorWrappedObsType = TypeVar("VectorWrappedObsType", default=Any)
"""The wrapped (inner) environment's batched observation type, for observation-transforming vector wrappers."""

VectorWrappedActType = TypeVar("VectorWrappedActType", default=Any)
"""The wrapped (inner) environment's batched action type, for action-transforming vector wrappers."""

VectorWrappedRewardType = TypeVar("VectorWrappedRewardType", default=Any)
"""The wrapped (inner) environment's batched reward type, for reward-transforming vector wrappers."""

# Deprecated: kept for backwards compatibility with downstream code that does
# `from gymnasium.vector.vector_env import ArrayType`. Prefer the dedicated
# reward/bool array type parameters above.
ArrayType = TypeVar("ArrayType", default=Any)
