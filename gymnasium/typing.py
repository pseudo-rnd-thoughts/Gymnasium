r"""Public typing vocabulary shared by Gymnasium's generic classes.

This module is the single, centralised source of the :class:`~typing.TypeVar`\\ s
used to parameterise Gymnasium's generic classes (precedent: :mod:`numpy.typing`).
Downstream code and Gymnasium's own modules should import these names from here
rather than redefining their own copies.

All of the TypeVars are **invariant** and declare ``default=Any`` (PEP 696), so a
class may be subscripted with as few or as many arguments as desired and any
omitted argument falls back to ``Any``.

The single-environment vocabulary parameterises
:class:`gymnasium.Env` ``[ObsType, ActType]`` and
:class:`gymnasium.Wrapper` ``[WrapperObsType, WrapperActType, ObsType, ActType]``;
the vector-environment vocabulary parameterises
:class:`gymnasium.vector.VectorEnv` ``[VectorObsType, VectorActType, RewardArrayType, BoolArrayType]``
and its wrappers. Each name's meaning is documented on the name itself below.
"""

from typing import Any, TypeAlias

import numpy as np
from typing_extensions import TypeVar

__all__ = [
    "ObsType",
    "ActType",
    "RenderFrame",
    "WrapperObsType",
    "WrapperActType",
    "VectorObsType",
    "VectorActType",
    "RewardArrayType",
    "BoolArrayType",
]

# Single-environment vocabulary
ObsType = TypeVar("ObsType", default=Any)
"""The observation type of an :class:`~gymnasium.Env`, i.e. what :meth:`~gymnasium.Env.reset` and :meth:`~gymnasium.Env.step` return and :attr:`~gymnasium.Env.observation_space` contains."""

ActType = TypeVar("ActType", default=Any)
"""The action type of an :class:`~gymnasium.Env`, i.e. what :meth:`~gymnasium.Env.step` accepts and :attr:`~gymnasium.Env.action_space` contains."""

RenderFrame: TypeAlias = str | np.ndarray | tuple[np.ndarray, np.ndarray]
"""A single frame returned by :meth:`~gymnasium.Env.render` (a concrete alias, not a TypeVar)."""

WrapperObsType = TypeVar("WrapperObsType", default=Any)
"""The observation type a :class:`~gymnasium.Wrapper` exposes to its user, possibly different from the wrapped environment's :data:`ObsType`."""

WrapperActType = TypeVar("WrapperActType", default=Any)
"""The action type a :class:`~gymnasium.Wrapper` accepts from its user, possibly different from the wrapped environment's :data:`ActType`."""

# Vector-environment vocabulary
VectorObsType = TypeVar("VectorObsType", default=Any)
"""The batched observation type of a :class:`~gymnasium.vector.VectorEnv`."""

VectorActType = TypeVar("VectorActType", default=Any)
"""The batched action type of a :class:`~gymnasium.vector.VectorEnv`."""

RewardArrayType = TypeVar("RewardArrayType", default=Any)
"""The batched reward array type of a :class:`~gymnasium.vector.VectorEnv`, typically ``np.ndarray`` of ``float64``."""

BoolArrayType = TypeVar("BoolArrayType", default=Any)
"""The batched termination/truncation array type of a :class:`~gymnasium.vector.VectorEnv`, typically ``np.ndarray`` of ``bool``."""
