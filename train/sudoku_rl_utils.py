"""Utility helpers for Sudoku specific RL fine-tuning.

The training pipeline introduced for Sudoku decoding has to orchestrate a
number of small numerical tricks (location distributions, log-sum-exp based
reductions, …).  Keeping these helpers in a dedicated module makes the main
training script considerably easier to read and reason about.

This module purposely stays agnostic to the Sudoku domain itself.  Problem
specific logic such as identifying deterministic locations is delegated to
``train.sudoku_tools`` which users can tailor to their own Sudoku
representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch


@dataclass
class DefinitePosition:
    """Container describing a position that can be deterministically filled."""

    index: int
    token_id: int


def normalise_definite_positions(
    definites: Iterable,
    mask_start: int,
    tokenizer,
) -> List[DefinitePosition]:
    """Normalise ``detect_definite`` outputs into :class:`DefinitePosition`.

    ``detect_definite`` is implemented outside of the repository and therefore
    its return type can not be enforced here.  In practice it is expected to
    return one of the following structures (or a homogeneous list of them):

    * ``[{"index": int, "token": int/str}]``
    * ``[(index, token)]``
    * ``[index, {"token": ...}]``

    ``index`` must be expressed relative to the first token after the prompt as
    specified in the project requirements.  The helper converts these relative
    indices into absolute token positions by adding ``mask_start``.
    """

    normalised: List[DefinitePosition] = []

    for item in definites:
        rel_index: Optional[int] = None
        token_obj: Optional[int] = None

        if isinstance(item, DefinitePosition):
            normalised.append(item)
            continue

        if isinstance(item, dict):
            rel_index = item.get("index")
            token_obj = item.get("token")
            if token_obj is None:
                token_obj = item.get("token_id")
        elif isinstance(item, (tuple, list)):
            if len(item) == 0:
                continue
            rel_index = int(item[0])
            if len(item) > 1:
                token_obj = item[1]
        elif isinstance(item, int):
            rel_index = int(item)
            token_obj = None
        else:
            raise TypeError(
                "Unsupported type returned by detect_definite: "
                f"{type(item)!r}. Please return ints, tuples or dicts."
            )

        if rel_index is None:
            raise ValueError(
                "detect_definite must provide an index for each definite "
                "position."
            )

        abs_index = mask_start + int(rel_index)

        if token_obj is None:
            raise ValueError(
                "detect_definite must provide the target token for each "
                "position.  Received an entry without token information."
            )

        if isinstance(token_obj, str):
            tokenised = tokenizer(
                token_obj,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            if isinstance(tokenised, dict):
                token_ids = tokenised.get("input_ids", [])
            else:
                token_ids = tokenised
            if len(token_ids) != 1:
                raise ValueError(
                    "Token strings returned by detect_definite must map to "
                    "exactly one tokenizer id."
                )
            token_id = int(token_ids[0])
        else:
            token_id = int(token_obj)

        normalised.append(DefinitePosition(index=abs_index, token_id=token_id))

    return normalised


def safe_logsumexp(
    log_terms: Sequence[torch.Tensor],
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return ``exp(logsumexp(log_terms))`` with gradient support."""

    if len(log_terms) == 0:
        target_device = device
        if target_device is None:
            target_device = torch.device("cpu")
        return torch.tensor(0.0, device=target_device)

    stack = torch.stack(log_terms)
    return torch.exp(torch.logsumexp(stack, dim=0))


def build_location_distribution(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    epsilon: float,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the distribution over decoding locations.

    The scoring scheme follows the requirement document: the margin between the
    top-1 and top-2 token probabilities is converted into a temperature-scaled
    softmax over candidate positions.
    """

    if logits.ndim != 2:
        raise ValueError("logits must be a (L, V) tensor")

    log_probs = torch.log_softmax(logits, dim=-1)
    top_log_probs, _ = torch.topk(log_probs, k=2, dim=-1)
    top_probs = torch.exp(top_log_probs)

    score = torch.log((top_probs[..., 0] + epsilon) / (top_probs[..., 1] + epsilon))

    masked_score = torch.full_like(score, float("-inf"))
    masked_score = torch.where(candidate_mask, score / max(temperature, 1e-8), masked_score)

    probs = torch.softmax(masked_score, dim=-1)
    return probs, masked_score


def build_token_distribution(
    logits: torch.Tensor,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return probability and log-probability for a token distribution."""

    scaled_logits = logits / max(temperature, 1e-8)
    log_probs = torch.log_softmax(scaled_logits, dim=-1)
    probs = torch.exp(log_probs)
    return probs, log_probs


def pad_to_length(sequence: Sequence[float], length: int) -> torch.Tensor:
    """Pad a python sequence with zeros so it can be stacked."""

    values = list(sequence)
    if len(values) >= length:
        return torch.tensor(values[:length], dtype=torch.float32)

    pad_amount = length - len(values)
    padded = values + [0.0] * pad_amount
    return torch.tensor(padded, dtype=torch.float32)


def compute_f_theta_value(
    logits: torch.Tensor,
    state_ids: torch.Tensor,
    *,
    mask_token_id: int,
    mask_start: int,
    tokenizer,
    lambda_coeff: float,
    epsilon: float,
    location_temperature: float,
    token_temperature: float,
    map_indices: Optional[torch.Tensor],
    detect_fn,
) -> torch.Tensor:
    """Compute :math:`f_\theta` for a decoded Sudoku state."""

    candidate_mask = state_ids.eq(mask_token_id)
    if not torch.any(candidate_mask):
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    location_probs, _ = build_location_distribution(
        logits,
        candidate_mask=candidate_mask,
        epsilon=epsilon,
        temperature=location_temperature,
    )

    token_log_probs = torch.log_softmax(
        logits / max(token_temperature, 1e-8), dim=-1
    )

    if map_indices is None or len(map_indices) == 0:
        map_index_set = {int(idx) for idx in torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist()}
    else:
        map_index_set = {int(idx) for idx in map_indices.tolist()}

    definites_raw = detect_fn(state_ids.tolist(), mask_start)
    definites = normalise_definite_positions(definites_raw, mask_start, tokenizer)

    s1_terms = []
    for position in definites:
        if position.index not in map_index_set:
            continue
        log_term = torch.log(location_probs[position.index] + 1e-12) + token_log_probs[
            position.index, position.token_id
        ]
        s1_terms.append(log_term)

    candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten().tolist()
    s2_terms = [
        torch.log(location_probs[idx] + 1e-12)
        for idx in candidate_indices
        if idx not in map_index_set
    ]

    s1_sum = safe_logsumexp(s1_terms, device=logits.device)
    s2_sum = safe_logsumexp(s2_terms, device=logits.device)

    return lambda_coeff * s2_sum + s1_sum
