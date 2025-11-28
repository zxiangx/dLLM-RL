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

import time
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import tqdm
from dataclasses import dataclass, field
from torch.utils.data import Dataset
import os
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple, cast
from train.sudoku_tools import detect_definite, pre_fill, judge_error
import json
from torch.utils.data import DataLoader
from models import LLaDAModelLM
from models.sampling import gumbel_sample
from collections import deque

from models.llada.modeling_llada import PromptCacheEntry
@dataclass
class DefinitePosition:
    """Container describing a position that can be deterministically filled."""

    index: int
    token_id: int

class SudokuPromptDataset(Dataset):
    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not locate dataset: {data_path}")

        records: List[Dict[str, Any]] = []
        if data_path.endswith(".jsonl"):
            with open(data_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
        else:
            raise ValueError(
                "Dataset should be jsonl file."
            )

        if not records:
            raise ValueError("Dataset is empty – at least one prompt is required")

        for record in records:
            record.setdefault("metadata", {})

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        return item


@dataclass
class StepRecord: # 用来保存一个step的详情
    sample_idx: int
    trajectory_idx: int
    step_index: int
    state_before: torch.Tensor # 两个都是token_ids状态
    location_index: int # 这一个step走了哪一个位置；以prompt后的第一个token为位置0
    token_id: int # 填入的token
    logprob_old_loc: float
    logprob_old_tok: float # 旧策略下的两个概率，用来后续计算ratio
    mask_start: int # prompt后第一个token位置
    map_indices: Optional[torch.Tensor] # 数独map的位置
    metadata: Dict[str, Any] = field(default_factory=dict) 

    advantage: float = 0.0
    f_theta_value: float = 0.0
    is_error: bool = False

    @property
    def old_logprob_sum(self) -> float:
        return self.logprob_old_loc + self.logprob_old_tok


@dataclass
class SamplingStats:
    entropy_sum: float = 0.0
    entropy_count: int = 0
    max_prob_sum: float = 0.0
    max_prob_count: int = 0
    greedy_inside_map: int = 0
    greedy_inside_s1: int = 0
    step_count: int = 0
    trajectory_count: int = 0
    error_terminated: int = 0
    reward_sum: float = 0.0

    def update_entropy(self, value: float) -> None:
        self.entropy_sum += float(value)
        self.entropy_count += 1

    def update_max_prob(self, value: float) -> None:
        self.max_prob_sum += float(value)
        self.max_prob_count += 1

    def update_greedy(self, in_map: bool, in_s1: bool) -> None:
        if in_map:
            self.greedy_inside_map += 1
            if in_s1:
                self.greedy_inside_s1 += 1

    def record_step(self, reward: float) -> None:
        self.step_count += 1
        self.reward_sum += float(reward)

    def finish_trajectory(self, error: bool) -> None:
        self.trajectory_count += 1
        if error:
            self.error_terminated += 1

    def merge(self, other: "SamplingStats") -> None:
        self.entropy_sum += other.entropy_sum
        self.entropy_count += other.entropy_count
        self.max_prob_sum += other.max_prob_sum
        self.max_prob_count += other.max_prob_count
        self.greedy_inside_map += other.greedy_inside_map
        self.greedy_inside_s1 += other.greedy_inside_s1
        self.step_count += other.step_count
        self.trajectory_count += other.trajectory_count
        self.error_terminated += other.error_terminated
        self.reward_sum += other.reward_sum


@dataclass
class PreparedSample:# 把prompt处理成适合rollout的格式
    state: torch.Tensor
    mask_start: int
    map_indices: torch.Tensor
    prompt_length: int
    map_range: Optional[Tuple[int, int]]
    metadata: Dict[str, Any]


@dataclass
class TrajectoryJob: # 记录一条轨迹的信息
    sample_idx: int # 问题编号
    trajectory_idx: int # 一个问题中的轨迹编号
    state: torch.Tensor # 问题初始状态
    mask_start: int # prompt长度
    map_indices: torch.Tensor
    map_index_set: set
    max_steps: int
    metadata: Dict[str, Any]
    records: List[StepRecord] = field(default_factory=list) # 记录了每一个step的信息
    steps_taken: int = 0
    terminated: bool = False

@dataclass
class ValidationJob:
    state: torch.Tensor
    max_steps: int
    mask_start: int
    steps_taken: int = 0

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
    # print(f"score.device: {score.device}, candidate_mask.device: {candidate_mask.device}, masked_score.device: {masked_score.device}, temperature.device: {torch.tensor(temperature).device}")
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


def prepare_sequence(
    sample: Dict[str, Any],
    tokenizer,
    mask_token_id: int,
    max_generation_length: int,
) -> PreparedSample:
    """transfer prompt for rollout"""
    prompt = sample["prompt"]

    filled_ids, prompt_length, map_range = pre_fill(
        prompt,
        tokenizer,
        max_generation_length,
    )

    filled_tensor = torch.tensor(filled_ids, dtype=torch.long)

    expected_total = prompt_length + max_generation_length
    if filled_tensor.size(0) != expected_total:
        raise ValueError(
            "pre_fill must return a sequence whose length equals prompt length"
            f" + max_generation_length ({expected_total}), got"
            f" {filled_tensor.size(0)}"
        )

    mask_positions = (filled_tensor == mask_token_id).nonzero(as_tuple=False).flatten()
    if mask_positions.numel() == 0:
        raise ValueError("Mask token never appears in the provided sequence")

    mask_start = int(prompt_length)

    if map_range is not None:
        map_start_rel, map_end_rel = map_range
        map_start_abs = mask_start + int(map_start_rel)
        map_end_abs = mask_start + int(map_end_rel)
        map_indices = torch.arange(map_start_abs, map_end_abs, dtype=torch.long)
        map_indices = map_indices[(map_indices >= 0) & (map_indices < filled_tensor.size(0))]
    else:
        map_indices = mask_positions.to(dtype=torch.long)

    if map_indices.numel() > 0:
        candidate_mask = filled_tensor.eq(mask_token_id)
        map_indices = map_indices[candidate_mask[map_indices]]

    metadata = sample.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {"value": metadata}

    return PreparedSample(
        state=filled_tensor,
        mask_start=mask_start,
        map_indices=map_indices.to(dtype=torch.long),
        prompt_length=prompt_length,
        map_range=map_range,
        metadata=metadata,
    )

def compute_definites(
    state_ids: torch.Tensor,
    mask_start: int,
    tokenizer,
) -> List[int]:
    from train.sudoku_rl_utils import normalise_definite_positions

    definites_raw = detect_definite(state_ids.detach().cpu().tolist(), mask_start)
    definites = normalise_definite_positions(definites_raw, mask_start, tokenizer)
    return [pos.index for pos in definites]


def save_checkpoint(
    model,
    tokenizer,
    accelerator: Accelerator,
    config,
    project_name: str,
    step: int|str,
    save_training_state: bool = True,
) -> None:
    """
    同时保存：
    1) HF 格式的 model + tokenizer（用于 from_pretrained 推理）
    2) Accelerate 的完整训练状态（用于 accelerator.load_state 断点续训）
    """

    step_label = f"step_{int(step):06d}" if isinstance(step, int) else str(step)
    ckpt_dir = Path(project_name) / "ckpt" / config.model.optimized_name / step_label

    # 目录创建
    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    # ===== 1. 保存 HuggingFace 模型 + tokenizer =====
    if accelerator.is_main_process:
        # 只在主进程 unwrap + get_state_dict，避免多卡重复干重活
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)

        # 保存为 HF 格式，兼容 from_pretrained
        unwrapped_model.save_pretrained(
            ckpt_dir,
            is_main_process=True,          # 现在只有 main 在干事
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,       # 保存成 safetensors
        )

        tokenizer.save_pretrained(str(ckpt_dir))

        # 额外写一点元信息（比如时间 / step）
        metadata = {
            "step": str(step),
            "save_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with (ckpt_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)

    accelerator.wait_for_everyone()

    # ===== 2. 保存 Accelerate 训练状态 =====
    # 这一步会在 ckpt_dir 下面再写 optimizer / RNG 等各种状态
    if save_training_state:
        accelerator.save_state(str(ckpt_dir))
        accelerator.wait_for_everyone()

def run_validation(
    accelerator: Accelerator,
    model: LLaDAModelLM,
    tokenizer,
    config,
    dataloader: Optional[DataLoader],
    global_step: int,
) -> None:
    if dataloader is None:
        return

    model.eval()
    training_cfg = config.training
    mask_token_id = training_cfg.mask_token_id
    rollout_batch_size = max(int(training_cfg.rollout_batch_size), 1)

    total_sequences = 0
    correct_sequences = 0
    total_decode_steps = 0

    device = accelerator.device

    def _finalise(job: ValidationJob) -> None:
        nonlocal total_sequences, correct_sequences
        total_sequences += 1
        is_correct = bool(not judge_error(job.state.detach().cpu().tolist(), job.mask_start))
        if is_correct:
            correct_sequences += 1

    with torch.no_grad():
        val_pbar = tqdm(
            total=len(dataloader),
            desc="Validation",
            disable=not accelerator.is_local_main_process,
            leave=False,
        )
                
        for batch in dataloader:
            prepared_samples = [
                prepare_sequence(
                    sample,
                    tokenizer,
                    mask_token_id,
                    training_cfg.max_generation_length,
                )
                for sample in batch
            ]

            job_queue: Deque[ValidationJob] = deque()

            for prepared in prepared_samples:
                mask_count = int((prepared.state == mask_token_id).sum().item())
                if mask_count <= 0:
                    job = ValidationJob(
                        state=prepared.state.clone(), 
                        mask_start=prepared.mask_start,
                        max_steps=0
                        )
                    _finalise(job)
                    continue

                job_queue.append(
                    ValidationJob(
                        state=prepared.state.clone().to(device),
                        mask_start=prepared.mask_start,
                        max_steps=mask_count,
                    )
                )

            while job_queue:
                active_jobs: List[ValidationJob] = []
                while job_queue and len(active_jobs) < rollout_batch_size:
                    active_jobs.append(job_queue.popleft())

                logits_batch = compute_logits_with_padding([job.state for job in active_jobs], model, tokenizer, accelerator.device)
                for batch_idx, job in enumerate(active_jobs):
                    logits = logits_batch[batch_idx].to(torch.float32)
                    candidate_mask = job.state.eq(mask_token_id)

                    if (not torch.any(candidate_mask)) or job.steps_taken >= job.max_steps:
                        _finalise(job)
                        continue

                    loc_probs, loc_scores = build_location_distribution(
                        logits,
                        candidate_mask=candidate_mask,
                        epsilon=training_cfg.epsilon_small,
                        temperature=training_cfg.location_temperature,
                    )

                    loc_logits = torch.where(
                        candidate_mask,
                        loc_scores,
                        torch.full_like(loc_scores, -1e9),
                    )
                    location_index = int(gumbel_sample(loc_logits).item())

                    token_logits_scaled = logits[location_index] / max(
                        training_cfg.token_temperature, 1e-8
                    )
                    token_id = int(gumbel_sample(token_logits_scaled).item())

                    next_state = job.state.clone()
                    next_state[location_index] = token_id

                    job.state = next_state
                    job.steps_taken += 1
                    total_decode_steps += 1

                    next_candidate_mask = job.state.eq(mask_token_id)
                    if (not torch.any(next_candidate_mask)) or job.steps_taken >= job.max_steps:
                        _finalise(job)
                    else:
                        job_queue.append(job)

            val_pbar.update(1)

        val_pbar.close()

    totals = torch.tensor(
        [total_sequences, correct_sequences, total_decode_steps],
        device=accelerator.device,
        dtype=torch.float64,
    )
    totals = accelerator.reduce(totals, reduction="sum")

    if accelerator.is_main_process:
        total = int(totals[0].item())
        correct = int(totals[1].item())
        decode_steps = float(totals[2].item())

        metrics: Dict[str, float] = {
            "val/num_samples": float(total),
            "val/num_correct": float(correct),
            "val/total_decode_steps": decode_steps,
        }

        if total > 0:
            metrics["val/accuracy"] = correct / total
            metrics["val/avg_decode_steps"] = decode_steps / max(total, 1)

        accelerator.log(metrics, step=global_step)

class PromptCacheManager:
    """Simple per-process prompt kv-cache store."""

    def __init__(self):
        self._cache: Dict[Tuple[int, ...], PromptCacheEntry] = {}

    def clear(self) -> None:
        self._cache.clear()

    def get(self, prompt_ids: torch.Tensor) -> Optional[PromptCacheEntry]:
        return self._cache.get(tuple(prompt_ids.tolist()))

    def add(self, prompt_ids: torch.Tensor, entry: PromptCacheEntry) -> PromptCacheEntry:
        self._cache[tuple(prompt_ids.tolist())] = entry
        return entry

    def __len__(self) -> int:
        return len(self._cache)
    
def compute_logits_with_padding(
    states,
    model,
    tokenizer,
    device,
    prompt_lengths=None,
    prompt_cache: Optional[PromptCacheManager] = None,
    dev: bool = False,
    allow_cache_in_grad: bool = True,
):
    lengths = [len(s) for s in states]
    batch_size = len(states)

    use_prompt_cache = prompt_cache is not None and prompt_lengths is not None

    # Prompt caching path batches sequences sharing cached prefixes for parallel reuse.
    if use_prompt_cache:
        logits_list: List[Optional[torch.Tensor]] = [None] * batch_size
        cached_entries: List[PromptCacheEntry] = []
        cached_gen_tokens: List[torch.Tensor] = []
        cached_indices: List[int] = []

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        for idx, (state, prompt_len) in enumerate(zip(states, prompt_lengths)):
            prompt_len = int(prompt_len)
            prompt_tokens = state[:prompt_len].to(device)
            gen_tokens = state[prompt_len:]

            cache_entry: Optional[PromptCacheEntry] = None
            if prompt_cache is not None:
                cache_entry = prompt_cache.get(prompt_tokens)

            record_cache = prompt_cache is not None and (allow_cache_in_grad or not torch.is_grad_enabled())
            if cache_entry is None and prompt_cache is not None:
                with torch.no_grad():
                    built = model.prefill_prompt_cache(state.unsqueeze(0), prompt_len)
                if record_cache:
                    cache_entry = prompt_cache.add(prompt_tokens, built)
                else:
                    cache_entry = built

            if cache_entry is not None and gen_tokens.numel() > 0:
                cached_entries.append(cache_entry.to(device))
                cached_gen_tokens.append(gen_tokens.to(device))
                cached_indices.append(idx)
            else:
                logits_list[idx] = model(state.unsqueeze(0).to(device)).logits.squeeze(0)

        if cached_indices:
            max_gen_len = max(t.numel() for t in cached_gen_tokens)
            input_ids_batch = torch.full(
                (len(cached_indices), max_gen_len),
                pad_id,
                dtype=torch.long,
                device=device,
            )
            for row, tokens in enumerate(cached_gen_tokens):
                input_ids_batch[row, : tokens.numel()] = tokens

            stacked_past: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx in range(len(cached_entries[0].past_key_values)):
                keys = torch.cat([entry.past_key_values[layer_idx][0] for entry in cached_entries], dim=0)
                values = torch.cat([entry.past_key_values[layer_idx][1] for entry in cached_entries], dim=0)
                stacked_past.append((keys, values))

            logits_batch = model(
                input_ids_batch,
                past_key_values=stacked_past,
                use_cache=False,
            ).logits

            for row, sample_idx in enumerate(cached_indices):
                state = states[sample_idx]
                prompt_len = int(prompt_lengths[sample_idx])
                gen_len = cached_gen_tokens[row].numel()
                full_logits = torch.zeros(
                    state.size(0), logits_batch.size(-1), device=device, dtype=logits_batch.dtype
                )
                full_logits[prompt_len : prompt_len + gen_len] = logits_batch[row, :gen_len]
                logits_list[sample_idx] = full_logits

        return cast(List[torch.Tensor], logits_list)
    max_len = max(lengths)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0 

    input_ids_batch = torch.full(
        (batch_size, max_len),
        pad_id,
        dtype=torch.long,
        device=device
    )

    for i, ids in enumerate(states):
        input_ids_batch[i, -len(ids):] = ids.to(device)

    if dev:
        print(f"Shapes:{input_ids_batch.size()}")
    logits_batch = model(input_ids_batch).logits  # (B, max_len, vocab_size)

    logits_list = []
    for i, orig_len in enumerate(lengths):
        logits_i = logits_batch[i, -orig_len:].clone()
        logits_list.append(logits_i)

    return logits_list
