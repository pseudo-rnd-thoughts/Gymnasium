from functools import partial
from typing import Callable

import pytest

from gymnasium import Space
from gymnasium.spaces import utils

TESTING_SPACE = Space()


@pytest.mark.parametrize(
    "func",
    [
        TESTING_SPACE.sample,
        partial(TESTING_SPACE.contains, None),
        partial(utils.flatdim, TESTING_SPACE),
        partial(utils.flatten, TESTING_SPACE, None),
        partial(utils.flatten_space, TESTING_SPACE),
        partial(utils.unflatten, TESTING_SPACE, None),
    ],
)
def test_not_implemented_errors(func: Callable[[], None]):
    with pytest.raises(NotImplementedError):
        func()
