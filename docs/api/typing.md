---
title: Typing
---

# Typing

```{eval-rst}
.. automodule:: gymnasium.typing
```

For example, a custom environment producing image observations and accepting discrete actions, and a wrapper that converts those observations to grayscale, would be annotated as:

```python
import numpy as np

import gymnasium as gym


class MyEnv(gym.Env[np.ndarray, int]):
    """An environment with `np.ndarray` observations and `int` actions."""


class GrayscaleWrapper[ActType](
    gym.ObservationWrapper[np.ndarray, ActType, np.ndarray]
):
    """Transforms `(H, W, 3)` uint8 observations into `(H, W)` grayscale ones."""

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.mean(observation, axis=-1).astype(np.uint8)
```

Note the [PEP 695](https://peps.python.org/pep-0695/) syntax: `GrayscaleWrapper` stays generic in its action type by declaring `[ActType]` as its own type parameter, rather than importing a shared `TypeVar`. Type parameters are lexically scoped to the class that declares them, so a name imported from `gymnasium.typing` cannot parameterise your class — it would be treated as a fixed type instead, silently making the class non-generic.

Every type parameter defaults to `Any` ([PEP 696](https://peps.python.org/pep-0696/)), so `gym.Env`, `gym.Wrapper[np.ndarray, int]` and other partial subscriptions remain valid.

## Single-environment vocabulary

```{eval-rst}
.. autodata:: gymnasium.typing.ObsType
   :no-value:
.. autodata:: gymnasium.typing.ActType
   :no-value:
.. autodata:: gymnasium.typing.RenderFrame
   :no-value:
.. autodata:: gymnasium.typing.WrapperObsType
   :no-value:
.. autodata:: gymnasium.typing.WrapperActType
   :no-value:
```

## Vector-environment vocabulary

```{eval-rst}
.. autodata:: gymnasium.typing.VectorObsType
   :no-value:
.. autodata:: gymnasium.typing.VectorActType
   :no-value:
.. autodata:: gymnasium.typing.VectorRewardType
   :no-value:
.. autodata:: gymnasium.typing.VectorBoolType
   :no-value:
.. autodata:: gymnasium.typing.VectorWrappedObsType
   :no-value:
.. autodata:: gymnasium.typing.VectorWrappedActType
   :no-value:
.. autodata:: gymnasium.typing.VectorWrappedRewardType
   :no-value:
```
