"""Hooks that need to be implemented for the Sudoku training pipeline.

The RL objective relies on three environment specific functions.  They are kept
in a dedicated module so that users can override them without having to touch
training code.  The default implementations raise :class:`NotImplementedError`
with detailed guidance which should make it straightforward to plug-in project
specific logic.
"""
from __future__ import annotations

from typing import Iterable, Sequence


def pre_fill(prompt: str, token_ids: Sequence[int]):
    """Fill deterministic tokens before sampling starts.

    Parameters
    ----------
    prompt:
        Human readable description of the Sudoku instance.
    token_ids:
        Tokenised representation of ``prompt`` including mask tokens that mark
        Sudoku slots.

    Returns
    -------
    filled_ids: list[int]
        Sequence with the deterministic values inserted.
    filled_count: int
        Number of tokens that were filled.
    filled_indices: Iterable[int]
        Indices (relative to the first token after the prompt) that were filled
        by this routine.
    """

    raise NotImplementedError(
        "Please provide a project specific implementation of pre_fill(prompt, "
        "token_ids).  The function must return (filled_ids, filled_count, "
        "filled_indices)."
    )


def detect_definite(state_ids: Sequence[int]):
    """Identify positions that can be filled with certainty.

    The function must return an iterable that is understood by
    :func:`train.sudoku_rl_utils.normalise_definite_positions`.  Returning a list
    of ``(index, token_id)`` tuples is the most convenient option.
    """

    raise NotImplementedError(
        "Please implement detect_definite(state_ids) and return the indices "
        "(relative to the first token after the prompt) together with their "
        "target token ids."
    )


def judge_error(state_ids: Sequence[int]) -> bool:
    """Return ``True`` if the partially decoded Sudoku violates any constraint."""

    raise NotImplementedError(
        "Please implement judge_error(state_ids) so that the training loop can "
        "terminate trajectories once an invalid Sudoku state is reached."
    )
