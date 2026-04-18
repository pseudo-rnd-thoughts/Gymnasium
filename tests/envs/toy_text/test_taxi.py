"""Tests for rainy Taxi stochastic transition logic.

Covers two bugs reported in the rainy-taxi variant:
  - #1509: Lateral moves are computed even when the primary N/S move is blocked
           by a grid boundary, inconsistent with the E/W wall-blocked behaviour.
  - #1510: Left/right are swapped for south and east headings, and N/S lateral
           moves are incorrectly gated by an east/west wall check.

The grid and its wall layout used in assertions:

    +---------+
    |R: | : :G|   row 0
    | : | : : |   row 1
    | : : : : |   row 2
    | | : | : |   row 3
    |Y| : |B: |   row 4
    +---------+
      0 1 2 3 4   col

Interior east walls (positions where moving east is blocked):
    (0,1)→(0,2), (1,1)→(1,2), (3,0)→(3,1), (3,2)→(3,3),
    (4,0)→(4,1), (4,2)→(4,3)
"""

import pytest

from gymnasium.envs.toy_text.taxi import TaxiEnv


def _lateral_positions(env, row, col, action, pass_idx=0, dest_idx=1):
    """Return the destination (row, col) for each of the 3 rainy transitions.

    Returns a tuple (ahead, left, right) matching the order in env.P.
    """
    state = env.encode(row, col, pass_idx, dest_idx)
    transitions = env.P[state][action]
    assert len(transitions) == 3, (
        f"Expected 3 rainy transitions, got {len(transitions)}"
    )
    return tuple(
        tuple(env.decode(new_state))[:2]  # (new_row, new_col)
        for _prob, new_state, _reward, _done in transitions
    )


@pytest.fixture
def rainy_env():
    return TaxiEnv(is_rainy=True)


def test_rainy_lateral_south_left_is_east_right_is_west(rainy_env):
    """Moving south: left neighbour is to the east, right is to the west."""
    ahead, left, right = _lateral_positions(rainy_env, row=1, col=3, action=0)
    assert ahead == (2, 3)
    assert left == (1, 4), f"left of south should be east (1, 4), got {left}"
    assert right == (1, 2), f"right of south should be west (1, 2), got {right}"


def test_rainy_lateral_north_left_is_west_right_is_east(rainy_env):
    """Moving north: left neighbour is to the west, right is to the east."""
    ahead, left, right = _lateral_positions(rainy_env, row=1, col=3, action=1)
    assert ahead == (0, 3)
    assert left == (1, 2), f"left of north should be west (1, 2), got {left}"
    assert right == (1, 4), f"right of north should be east (1, 4), got {right}"


def test_rainy_lateral_east_left_is_north_right_is_south(rainy_env):
    """Moving east: left neighbour is to the north, right is to the south."""
    ahead, left, right = _lateral_positions(rainy_env, row=1, col=3, action=2)
    assert ahead == (1, 4)
    assert left == (0, 3), f"left of east should be north (0, 3), got {left}"
    assert right == (2, 3), f"right of east should be south (2, 3), got {right}"


def test_rainy_lateral_west_left_is_south_right_is_north(rainy_env):
    """Moving west: left neighbour is to the south, right is to the north."""
    ahead, left, right = _lateral_positions(rainy_env, row=1, col=3, action=3)
    assert ahead == (1, 2)
    assert left == (2, 3), f"left of west should be south (2, 3), got {left}"
    assert right == (0, 3), f"right of west should be north (0, 3), got {right}"


# ---------------------------------------------------------------------------
# Issue #1510 (wall variant) — N/S lateral moves must not be gated by
# east/west wall checks at the destination cell.
#
# When the primary action is east or west, the two lateral moves are
# north and south.  North/south movement has no walls; only the grid
# boundary can prevent it.  The offset-based wall check used for E/W
# movement must not be applied here.
# ---------------------------------------------------------------------------


def test_rainy_west_south_lateral_not_blocked_at_right_edge(rainy_env):
    """Taxi at (1, 4) moving west: the southward lateral to (2, 4) must be open.

    The rightmost column has an outer wall at desc[*, 10].  An incorrect
    east/west wall check on a south movement reads that boundary character
    and spuriously blocks the move.
    """
    ahead, left, right = _lateral_positions(rainy_env, row=1, col=4, action=3)
    assert ahead == (1, 3)
    # left of west = south; (2, 4) is a valid cell with no wall between rows
    assert left == (2, 4), (
        f"south lateral from (1, 4) going west should reach (2, 4), got {left}"
    )
    assert right == (0, 4)


def test_rainy_east_north_lateral_not_blocked_by_upper_east_wall(rainy_env):
    """Taxi at (2, 1) moving east: the northward lateral to (1, 1) must be open.

    There is an east wall between (1, 1) and (1, 2), recorded in the desc as
    desc[2, 4] = '|'.  An incorrect wall check reads this character when
    testing whether the taxi can move north to (1, 1), and wrongly blocks it.
    """
    ahead, left, right = _lateral_positions(rainy_env, row=2, col=1, action=2)
    assert ahead == (2, 2)
    # left of east = north; row boundary is not an issue here
    assert left == (1, 1), (
        f"north lateral from (2, 1) going east should reach (1, 1), got {left}"
    )
    # right of east = south
    assert right == (3, 1), (
        f"south lateral from (2, 1) going east should reach (3, 1), got {right}"
    )


# ---------------------------------------------------------------------------
# Issue #1509 — When the primary move is impossible, no lateral drift
#
# East/west primary moves blocked by an interior wall produce all three
# transitions staying at the current cell (the if-block is skipped entirely).
# North/south primary moves blocked by the grid boundary must behave the
# same way: no lateral movement should occur.
# ---------------------------------------------------------------------------


def test_rainy_south_at_bottom_boundary_no_lateral_drift(rainy_env):
    """Taxi at row 4 (bottom boundary) moving south: all outcomes stay put.

    The primary south move is blocked by the grid edge.  Neither left nor
    right lateral moves should be computed, matching the wall-blocked E/W
    behaviour.
    """
    ahead, left, right = _lateral_positions(rainy_env, row=4, col=3, action=0)
    assert ahead == (4, 3)
    assert left == (4, 3), (
        f"blocked south at boundary should not drift; left went to {left}"
    )
    assert right == (4, 3), (
        f"blocked south at boundary should not drift; right went to {right}"
    )


def test_rainy_north_at_top_boundary_no_lateral_drift(rainy_env):
    """Taxi at row 0 (top boundary) moving north: all outcomes stay put.

    Same consistency requirement as the south-boundary case above.
    """
    ahead, left, right = _lateral_positions(rainy_env, row=0, col=3, action=1)
    assert ahead == (0, 3)
    assert left == (0, 3), (
        f"blocked north at boundary should not drift; left went to {left}"
    )
    assert right == (0, 3), (
        f"blocked north at boundary should not drift; right went to {right}"
    )
