import os
import json
import sys
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import tqdm
from accelerate.utils import set_seed, gather_object
from omegaconf import OmegaConf
import time
import wandb
from transformers import AutoTokenizer
from math_verify import parse as math_parse, verify as math_verify

from models import LLaDAModelLM
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_error, set_verbosity_info
from models.sampling import gumbel_sample, gumbel_topk

from train.utils import flatten_omega_conf, get_config
from train.sudoku_rl_utils import (
    build_location_distribution,
    build_token_distribution,
    compute_logits_with_padding,
    PromptCacheManager,
    save_checkpoint,
)
from train.rl_math_utils import (
    MathPromptDataset,
    CompletedTrajectory,
    StepRecord,
    SamplingStats,
    PreparedSample,
    TrajectoryJob,
    SampleTrajectoryBundle,
    extract_final_boxed_answer,
    prepare_sequence
)
logger = get_logger(__name__, log_level="INFO")


def collect_rollouts(
    batch: Sequence[Dict[str, Any]],
    model: LLaDAModelLM,
    tokenizer,
    config,
    accelerator: Accelerator,
    progress_desc: str = "",
    enable_bar: bool = False,
    prompt_cache: Optional[PromptCacheManager] = None,
) -> Tuple[List[StepRecord], List[CompletedTrajectory], SamplingStats]:
    training_cfg = config.training
    mask_token_id = training_cfg.mask_token_id
    device = accelerator.device

    prepared_samples: List[PreparedSample] = [
        prepare_sequence(sample, tokenizer, mask_token_id, training_cfg.max_generation_length)
        for sample in batch
    ]

    job_queue: Deque[TrajectoryJob] = deque()
    for sample_idx, prepared in enumerate(prepared_samples):
        for group_idx in range(training_cfg.group_size):
            job_queue.append(
                TrajectoryJob(
                    sample_idx=sample_idx,
                    trajectory_idx=group_idx,
                    state=prepared.state.clone().to(device),
                    prompt=prepared.prompt,
                    prompt_length=prepared.prompt_length,
                    max_steps=training_cfg.max_generation_length,
                    metadata=prepared.metadata,
                    answer=prepared.answer,
                )
            )

    stats = SamplingStats()
    completed: List[CompletedTrajectory] = []
    all_records: List[StepRecord] = []

    total_masks = len(job_queue) * training_cfg.max_generation_length
    rollout_pbar = None
    if total_masks > 0 and enable_bar:
        desc = progress_desc or "Rollout"
        rollout_pbar = tqdm(
            total=total_masks,
            desc=desc,
            leave=False,
            disable=False,
            main_process_only=False,
            position=accelerator.process_index,
        )

    rollout_batch_size = max(int(training_cfg.rollout_batch_size), 1)
    tokens_per_step = max(int(training_cfg.get("tokens_per_step", 1)), 1)
    steps_finished = 0

    while job_queue:
        active_jobs: List[TrajectoryJob] = []
        while job_queue and len(active_jobs) < rollout_batch_size:
            active_jobs.append(job_queue.popleft())

        with torch.no_grad():
            logits_batch = compute_logits_with_padding(
                [job.state for job in active_jobs],
                model,
                tokenizer,
                accelerator.device,
                prompt_lengths=[job.prompt_length for job in active_jobs],
                prompt_cache=prompt_cache,
            )

        for batch_idx, job in enumerate(active_jobs):
            logits = logits_batch[batch_idx].to(torch.float32)
            candidate_mask = job.state.eq(mask_token_id)
            if not torch.any(candidate_mask):
                completed.append(
                    CompletedTrajectory(
                        sample_idx=job.sample_idx,
                        trajectory_idx=job.trajectory_idx,
                        final_state=job.state.detach().cpu(),
                        prompt_length=job.prompt_length,
                        prompt=job.prompt,
                        metadata=job.metadata,
                        answer=job.answer,
                    )
                )
                continue

            loc_probs, loc_scores = build_location_distribution(
                logits,
                candidate_mask=candidate_mask,
                epsilon=training_cfg.epsilon_small,
                temperature=training_cfg.location_temperature,
            )

            valid_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            valid_probs = loc_probs[valid_indices]
            entropy = -(valid_probs * torch.log(valid_probs + 1e-12)).sum()
            stats.update_entropy(float(entropy))
            stats.update_max_prob(float(valid_probs.max()))

            loc_logits = torch.where(candidate_mask, loc_scores, torch.full_like(loc_scores, -1e9))
            tokens_to_decode = min(tokens_per_step, int(candidate_mask.sum().item()))
            selected_locations = gumbel_topk(loc_logits, tokens_to_decode)

            next_state = job.state.clone()
            for location_index_tensor in selected_locations:
                location_index = int(location_index_tensor.item())
                logprob_loc = float(torch.log(loc_probs[location_index] + 1e-12))

                token_logits_scaled = logits[location_index] / max(
                    training_cfg.token_temperature, 1e-8
                )
                token_id = int(gumbel_sample(token_logits_scaled).item())
                token_log_probs = torch.log_softmax(token_logits_scaled, dim=-1)
                logprob_tok = float(token_log_probs[token_id])

                next_state[location_index] = token_id

                step_record = StepRecord(
                    sample_idx=job.sample_idx,
                    trajectory_idx=job.trajectory_idx,
                    step_index=job.steps_taken,
                    state_before=job.state.detach().cpu(),
                    location_index=location_index,
                    token_id=token_id,
                    logprob_old_loc=logprob_loc,
                    logprob_old_tok=logprob_tok,
                    prompt_length=job.prompt_length,
                    metadata=job.metadata,
                )

                all_records.append(step_record)

            job.state = next_state
            job.steps_taken += 1

            if rollout_pbar is not None:
                steps_finished += tokens_to_decode
                rollout_pbar.update(tokens_to_decode)

            job_queue.append(job)

    if rollout_pbar is not None:
        rollout_pbar.close()

    return all_records, completed, stats


def _normalise_rewards(rewards: List[float]) -> List[float]:
    if not rewards:
        return []
    tensor = torch.tensor(rewards, dtype=torch.float32)
    mean = tensor.mean()
    std = tensor.std(unbiased=False)
    if std < 1e-6:
        return [0.0 for _ in rewards]
    return ((tensor - mean) / std).tolist()


def _decode_answer(tokenizer, state: torch.Tensor, prompt_length: int) -> str:
    answer_tokens = state[prompt_length:]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()


def _extract_boxed_answer(text: str) -> Tuple[str, bool]:
    extracted = extract_final_boxed_answer(text)
    success = extracted != "Can not extract the answer!"
    return extracted.strip(), success


def _is_answer_correct(output: str, reference: str) -> bool:
    if not reference:
        return False

    extracted_output, format_ok = _extract_boxed_answer(output)
    if not format_ok:
        return False

    try:
        parsed_output = math_parse(extracted_output)
        parsed_ref = math_parse(reference)
        return bool(math_verify(parsed_output, parsed_ref))
    except Exception:
        return False


def _math_reward(output: str, reference: str) -> float:
    extracted_output, format_ok = _extract_boxed_answer(output)
    format_reward = 1.0 if format_ok else 0.0

    if not format_ok or not reference:
        return format_reward

    correct_reward = 0.0
    try:
        parsed_output = math_parse(extracted_output)
        parsed_ref = math_parse(reference)
        correct_reward = 1.0 if math_verify(parsed_output, parsed_ref) else 0.0
    except Exception:
        correct_reward = 0.0

    return format_reward + correct_reward

def _store_debug_rollouts(
    completed: List[CompletedTrajectory],
    step_records: List[StepRecord],
    tokenizer,
    store_path: str,
    accelerator: Accelerator,
) -> None:
    if not completed or not store_path:
        return

    root = Path(store_path)
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / f"{accelerator.process_index}.jsonl"

    advantage_map: Dict[Tuple[int, int], float] = {}
    for step in step_records:
        advantage_map[(step.sample_idx, step.trajectory_idx)] = float(step.advantage)

    with open(out_path, "a", encoding="utf-8") as fh:
        for traj in completed:
            output_text = _decode_answer(tokenizer, traj.final_state, traj.prompt_length)
            is_true = _is_answer_correct(output_text, traj.answer)
            record = {
                "prompt": getattr(traj, "prompt", ""),
                "answer": traj.answer,
                "test_results": output_text,
                "is_true": is_true,
                "advantage": advantage_map.get(
                    (traj.sample_idx, traj.trajectory_idx), None
                ),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def assign_advantages(
    step_records: List[StepRecord],
    completed: List[CompletedTrajectory],
    tokenizer,
    group_size: int,
) -> float:
    """Assign normalised group rewards as advantages."""

    # Organise completed trajectories by sample
    per_sample: Dict[int, List[CompletedTrajectory]] = {}
    for traj in completed:
        per_sample.setdefault(traj.sample_idx, []).append(traj)

    total_raw_reward = 0.0
    for sample_idx, trajs in per_sample.items():
        if not trajs:
            continue
        rewards: List[float] = []
        decoded_outputs: List[str] = []
        for traj in trajs:
            output_text = _decode_answer(tokenizer, traj.final_state, traj.prompt_length)
            decoded_outputs.append(output_text)
            rewards.append(_math_reward(output_text, traj.answer))
            total_raw_reward += rewards[-1]

        norm_rewards = _normalise_rewards(rewards)

        # map (trajectory_idx -> norm_reward)
        reward_map = {traj.trajectory_idx: norm for traj, norm in zip(trajs, norm_rewards)}

        for step in step_records:
            if step.sample_idx != sample_idx:
                continue
            step.advantage = float(reward_map.get(step.trajectory_idx, 0.0))

    return total_raw_reward


def _filter_sample_trajectories(
    step_records: List[StepRecord], completed: List[CompletedTrajectory], tokenizer, if_filter: bool=False
) -> Tuple[List[SampleTrajectoryBundle], List[bool]]:
    per_sample_steps: Dict[int, List[StepRecord]] = defaultdict(list)
    per_sample_completed: Dict[int, List[CompletedTrajectory]] = defaultdict(list)

    for step in step_records:
        per_sample_steps[step.sample_idx].append(step)

    for traj in completed:
        per_sample_completed[traj.sample_idx].append(traj)

    filtered: List[SampleTrajectoryBundle] = []
    missing_signals: List[bool] = []
    all_correct = 0
    all_error = 0
    for sample_idx, trajs in per_sample_completed.items():
        steps = per_sample_steps.get(sample_idx, [])
        if not trajs or not steps:
            missing_signals.append(True)
            continue

        rewards = [
            _math_reward(_decode_answer(tokenizer, traj.final_state, traj.prompt_length), traj.answer)
            for traj in trajs
        ]
        has_correct = any(reward >= 2.0 for reward in rewards)
        has_incorrect = any(reward < 2.0 for reward in rewards)
        missing_signals.append(not (has_correct and has_incorrect))
        if not has_correct: all_error+=1
        if not has_incorrect: all_correct+=1
        if not (has_correct and has_incorrect) and if_filter:
            continue

        filtered.append(SampleTrajectoryBundle(step_records=steps, completed=trajs))

    return filtered, missing_signals, all_correct, all_error


def _reindex_sample_trajectories(
    trajectories: List[SampleTrajectoryBundle], start_index: int = 0
) -> int:
    """对list中的bundle从0开始编号"""
    next_index = start_index
    for bundle in trajectories:
        for step in bundle.step_records:
            step.sample_idx = next_index
        for traj in bundle.completed:
            traj.sample_idx = next_index
        next_index += 1
    return next_index


# --- Loss computation -------------------------------------------------------


def compute_losses(
    step_records: List[StepRecord],
    model: LLaDAModelLM,
    tokenizer,
    config,
    accelerator: Accelerator,
    optimizer,
    lr_scheduler,
    progress_desc: str = "",
    enable_bar: bool = False,
    prompt_cache: Optional[PromptCacheManager] = None,
    collect_ratio_debug: bool = False,
) -> Tuple[float, float, float, int, Dict[str, List[Any]]]:
    device = accelerator.device
    if not step_records:
        return 0.0, 0.0, 0.0, {"locate_ratio": [], "token_ratio": [], "trajectory_ratio": []}

    training_cfg = config.training
    mask_token_id = training_cfg.mask_token_id
    micro_batch_size = max(int(training_cfg.micro_batch_size), 1)
    total_steps = len(step_records)

    grouped_steps: List[List[StepRecord]] = []
    grouped_map: "OrderedDict[Tuple[int, int, int], List[StepRecord]]" = OrderedDict()
    for step in step_records:
        key = (step.sample_idx, step.trajectory_idx, step.step_index)
        if key not in grouped_map:
            grouped_map[key] = []
            grouped_steps.append(grouped_map[key])
        grouped_map[key].append(step)

    rl_loss_sum = torch.tensor(0.0, device=device)
    rl_count = 0
    clip_events_tok = 0
    clip_events_loc = 0
    ratio_debug: Dict[str, List[Any]] = {
        "locate_ratio": [],
        "token_ratio": [],
        "trajectory_ratio": [],
    }

    traj_logprob_new_combined: Dict[Tuple[int, int], torch.Tensor] = {}
    traj_logprob_old_combined: Dict[Tuple[int, int], torch.Tensor] = {}
    traj_logprob_new_tok: Dict[Tuple[int, int], torch.Tensor] = {}
    traj_logprob_old_tok: Dict[Tuple[int, int], torch.Tensor] = {}
    traj_logprob_new_loc: Dict[Tuple[int, int], torch.Tensor] = {}
    traj_logprob_old_loc: Dict[Tuple[int, int], torch.Tensor] = {}
    traj_counts: Dict[Tuple[int, int], int] = {}

    num_forward_batches = math.ceil(len(grouped_steps) / micro_batch_size)
    loss_pbar = None
    if num_forward_batches > 0 and enable_bar:
        desc = progress_desc or "ComputeLoss"
        loss_pbar = tqdm(
            total=num_forward_batches,
            desc=desc,
            leave=False,
            disable=False,
            main_process_only=False,
        )

    for start in range(0, len(grouped_steps), micro_batch_size):
        with accelerator.accumulate(model):
            loop_start_time = time.time()
            batch_groups = grouped_steps[start : start + micro_batch_size]
            states_before = [group[0].state_before.to(device) for group in batch_groups]

            forward_start_time = time.time()
            logits_before = compute_logits_with_padding(
                states_before,
                model,
                tokenizer,
                accelerator.device,
                prompt_lengths=[group[0].prompt_length for group in batch_groups],
                prompt_cache=prompt_cache,
                allow_cache_in_grad=True,
            )
            forward_time = time.time() - forward_start_time

            if loss_pbar is not None:
                loss_pbar.update(1)

            combined_losses = []
            loss_compute_start_time = time.time()
            for idx, group in enumerate(batch_groups):
                logits_b = logits_before[idx]
                state_before = states_before[idx]
                candidate_mask = state_before.eq(mask_token_id)
                if not torch.any(candidate_mask):
                    continue

                loc_probs, _ = build_location_distribution(
                    logits_b,
                    candidate_mask=candidate_mask,
                    epsilon=training_cfg.epsilon_small,
                    temperature=training_cfg.location_temperature,
                )
                for step in group:
                    loc_prob_selected = loc_probs[step.location_index]
                    logprob_new_loc = torch.log(loc_prob_selected + 1e-12)

                    _, token_log_probs = build_token_distribution(
                        logits_b[step.location_index],
                        temperature=training_cfg.token_temperature,
                    )
                    logprob_new_tok = token_log_probs[step.token_id]

                    logprob_old_loc = torch.tensor(
                        step.logprob_old_loc, device=device, dtype=logprob_new_loc.dtype
                    )
                    logprob_old_tok = torch.tensor(
                        step.logprob_old_tok, device=device, dtype=logprob_new_tok.dtype
                    )

                    logprob_diff_loc = logprob_new_loc - logprob_old_loc
                    logprob_diff_tok = logprob_new_tok - logprob_old_tok

                    ratio_loc = torch.exp(logprob_diff_loc)
                    ratio_tok = torch.exp(logprob_diff_tok)

                    if collect_ratio_debug:
                        ratio_debug["locate_ratio"].append(
                            (
                                float(logprob_diff_loc.detach().cpu()),
                                float(logprob_new_loc.detach().cpu()),
                                float(logprob_old_loc.detach().cpu()),
                            )
                        )
                        ratio_debug["token_ratio"].append(
                            (
                                float(logprob_diff_tok.detach().cpu()),
                                float(logprob_new_tok.detach().cpu()),
                                float(logprob_old_tok.detach().cpu()),
                            )
                        )
                        traj_key = (step.sample_idx, step.trajectory_idx)
                        traj_logprob_new_combined[traj_key] = (
                            traj_logprob_new_combined.get(
                                traj_key, torch.tensor(0.0, device=device)
                            )
                            + (logprob_new_loc + logprob_new_tok)
                        )
                        traj_logprob_old_combined[traj_key] = (
                            traj_logprob_old_combined.get(
                                traj_key, torch.tensor(0.0, device=device)
                            )
                            + (logprob_old_loc + logprob_old_tok)
                        )

                        traj_logprob_new_tok[traj_key] = traj_logprob_new_tok.get(
                            traj_key, torch.tensor(0.0, device=device)
                        ) + logprob_new_tok
                        traj_logprob_old_tok[traj_key] = traj_logprob_old_tok.get(
                            traj_key, torch.tensor(0.0, device=device)
                        ) + logprob_old_tok

                        traj_logprob_new_loc[traj_key] = traj_logprob_new_loc.get(
                            traj_key, torch.tensor(0.0, device=device)
                        ) + logprob_new_loc
                        traj_logprob_old_loc[traj_key] = traj_logprob_old_loc.get(
                            traj_key, torch.tensor(0.0, device=device)
                        ) + logprob_old_loc

                        traj_counts[traj_key] = traj_counts.get(traj_key, 0) + 1

                    clip_ratio_tok = torch.clamp(
                        ratio_tok, 1 - training_cfg.token_clip_epsilon_low, 1 + training_cfg.token_clip_epsilon_high
                    )

                    clip_ratio_loc = torch.clamp(
                        ratio_loc, 1 - training_cfg.locate_clip_epsilon_low, 1 + training_cfg.locate_clip_epsilon_high
                    )

                    if (clip_ratio_tok - ratio_tok).abs() > 1e-8:
                        clip_events_tok += 1

                    if (clip_ratio_loc - ratio_loc).abs() > 1e-8:
                        clip_events_loc += 1

                    if training_cfg.ablation:
                        clip_ratio_loc = 1.0
                    adv = torch.tensor(step.advantage, device=device, dtype=clip_ratio_tok.dtype)
                    rl_term = -torch.min(ratio_tok * clip_ratio_loc * adv, clip_ratio_tok * clip_ratio_loc * adv)
                    # ratio_loc必须要clip
                    rl_loss_sum = rl_loss_sum + rl_term.detach()
                    rl_count += 1

                    combined_losses.append(rl_term / max(1, total_steps))

            total_loss = torch.tensor(0.0, device=device)
            for loss in combined_losses:
                total_loss += loss
            loss_compute_time = time.time() - loss_compute_start_time

            backward_start_time = time.time()
            accelerator.backward(total_loss)
            backward_time = time.time() - backward_start_time

            loop_total_time = time.time() - loop_start_time
            # print(
            #     f"[rank {accelerator.process_index}] compute_losses micro-batch timing - "
            #     f"forward: {forward_time:.2f}s, loss: {loss_compute_time:.2f}s, "
            #     f"backward: {backward_time:.2f}s, total: {loop_total_time:.2f}s",
            #     flush=True,
            # )
    if collect_ratio_debug:
        for traj_key, count in traj_counts.items():
            combined_new_sum = traj_logprob_new_combined.get(
                traj_key, torch.tensor(0.0, device=device)
            )
            combined_old_sum = traj_logprob_old_combined.get(
                traj_key, torch.tensor(0.0, device=device, dtype=combined_new_sum.dtype)
            )
            tok_new_sum = traj_logprob_new_tok.get(
                traj_key, torch.tensor(0.0, device=device, dtype=combined_new_sum.dtype)
            )
            tok_old_sum = traj_logprob_old_tok.get(
                traj_key, torch.tensor(0.0, device=device, dtype=combined_new_sum.dtype)
            )
            loc_new_sum = traj_logprob_new_loc.get(
                traj_key, torch.tensor(0.0, device=device, dtype=combined_new_sum.dtype)
            )
            loc_old_sum = traj_logprob_old_loc.get(
                traj_key, torch.tensor(0.0, device=device, dtype=combined_new_sum.dtype)
            )

            combined_new_avg = combined_new_sum / max(count, 1)
            combined_old_avg = combined_old_sum / max(count, 1)
            tok_new_avg = tok_new_sum / max(count, 1)
            tok_old_avg = tok_old_sum / max(count, 1)
            loc_new_avg = loc_new_sum / max(count, 1)
            loc_old_avg = loc_old_sum / max(count, 1)
            ratio_debug["trajectory_ratio"].append(
                {
                    "combined": (
                        float((combined_new_avg - combined_old_avg).detach().cpu()),
                        float(combined_new_avg.detach().cpu()),
                        float(combined_old_avg.detach().cpu()),
                    ),
                    "token_ratio": (
                        float((tok_new_avg - tok_old_avg).detach().cpu()),
                        float(tok_new_avg.detach().cpu()),
                        float(tok_old_avg.detach().cpu()),
                    ),
                    "locate_ratio": (
                        float((loc_new_avg - loc_old_avg).detach().cpu()),
                        float(loc_new_avg.detach().cpu()),
                        float(loc_old_avg.detach().cpu()),
                    ),
                }
            )

    if accelerator.sync_gradients:
        if training_cfg.max_grad_norm is not None:
            accelerator.clip_grad_norm_(model.parameters(), training_cfg.max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    if loss_pbar is not None:
        loss_pbar.close()

    rl_loss_mean = float((rl_loss_sum / max(rl_count, 1)).item())
    clip_fraction = (clip_events_tok / max(rl_count, 1), clip_events_loc / max(rl_count, 1))

    return rl_loss_mean, clip_fraction, float(rl_count), ratio_debug


def _decode_and_score(
    completed: List[CompletedTrajectory], tokenizer
) -> Tuple[int, int, float]:
    per_sample: Dict[int, List[CompletedTrajectory]] = {}
    for traj in completed:
        per_sample.setdefault(traj.sample_idx, []).append(traj)

    total_samples = 0
    correct_samples = 0
    reward_sum = 0.0

    for _, trajs in per_sample.items():
        for traj in trajs:
            total_samples += 1
            output_text = _decode_answer(tokenizer, traj.final_state, traj.prompt_length)
            reward = _math_reward(output_text, traj.answer)
            reward_sum += reward
            correct_samples += 1 if reward == 2.0 else 0

    return total_samples, correct_samples, reward_sum


def run_evaluation(
    accelerator: Accelerator,
    model: LLaDAModelLM,
    tokenizer,
    config,
    dataloader: Optional[DataLoader],
    global_step: int,
    tag: str,
) -> None:
    if dataloader is None:
        return

    model.eval()
    training_cfg = config.training

    total_samples = 0
    correct_samples = 0
    reward_sum = 0.0

    eval_group_size = int(training_cfg.get("evaluation_group_size", 1))
    eval_tokens_per_step = max(int(training_cfg.get("evaluation_tokens_per_step", 1)), 1)
    with torch.no_grad():
        eval_pbar = tqdm(
            iterable=enumerate(dataloader),
            total=len(dataloader),
            desc=f"{tag.title()}",
            disable=not accelerator.is_local_main_process,
            leave=False,
        )

        for batch_idx, batch in eval_pbar:
            prepared_samples = [
                prepare_sequence(
                    sample,
                    tokenizer,
                    training_cfg.mask_token_id,
                    training_cfg.max_generation_length,
                )
                for sample in batch
            ]

            job_queue: Deque[TrajectoryJob] = deque()
            for sample_idx, prepared in enumerate(prepared_samples):
                for group_idx in range(eval_group_size):
                    job_queue.append(
                        TrajectoryJob(
                            sample_idx=sample_idx,
                            trajectory_idx=group_idx,
                            state=prepared.state.clone().to(accelerator.device),
                            prompt_length=prepared.prompt_length,
                            prompt=prepared.prompt,
                            max_steps=training_cfg.max_generation_length,
                            metadata=prepared.metadata,
                            answer=prepared.answer,
                        )
                    )

            completed: List[CompletedTrajectory] = []

            while job_queue:
                active_jobs: List[TrajectoryJob] = []
                while job_queue and len(active_jobs) < training_cfg.rollout_batch_size:
                    active_jobs.append(job_queue.popleft())

                logits_batch = compute_logits_with_padding(
                    [job.state for job in active_jobs],
                    model,
                    tokenizer,
                    accelerator.device,
                )

                for batch_idx, job in enumerate(active_jobs):
                    logits = logits_batch[batch_idx].to(torch.float32)
                    candidate_mask = job.state.eq(training_cfg.mask_token_id)

                    if not torch.any(candidate_mask):
                        completed.append(
                            CompletedTrajectory(
                                sample_idx=job.sample_idx,
                                trajectory_idx=job.trajectory_idx,
                                final_state=job.state.detach().cpu(),
                                prompt_length=job.prompt_length,
                                prompt=job.prompt,
                                metadata=job.metadata,
                                answer=job.answer,
                            )
                        )
                        continue

                    loc_probs, loc_scores = build_location_distribution(
                        logits,
                        candidate_mask=candidate_mask,
                        epsilon=training_cfg.epsilon_small,
                        temperature=training_cfg.eval_location_temperature,
                    )

                    loc_logits = torch.where(
                        candidate_mask, loc_scores, torch.full_like(loc_scores, -1e9)
                    )
                    tokens_to_decode = min(
                        eval_tokens_per_step, int(candidate_mask.sum().item())
                    )
                    selected_locations = gumbel_topk(loc_logits, tokens_to_decode)

                    next_state = job.state.clone()
                    for location_index_tensor in selected_locations:
                        location_index = int(location_index_tensor.item())

                        token_logits_scaled = logits[location_index] / max(
                            training_cfg.eval_token_temperature, 1e-8
                        )
                        token_id = int(gumbel_sample(token_logits_scaled).item())

                        next_state[location_index] = token_id

                    job.state = next_state
                    job.steps_taken += 1

                    job_queue.append(job)

            batch_total, batch_correct, batch_reward = _decode_and_score(
                completed, tokenizer
            )

            total_samples += batch_total # trajectory count
            correct_samples += batch_correct # correct traj count
            reward_sum += batch_reward # total reward

            eval_pbar.update(1)

        eval_pbar.close()

    totals_tensor = torch.tensor(
        [total_samples, correct_samples, reward_sum],
        device=accelerator.device,
        dtype=torch.float64,
    )
    totals_tensor = accelerator.reduce(totals_tensor, reduction="sum")

    if accelerator.is_main_process:
        total = int(totals_tensor[0].item())
        correct = int(totals_tensor[1].item())
        reward_total = float(totals_tensor[2].item())

        metrics: Dict[str, float] = {
            f"{tag}/num_samples": float(total),
            f"{tag}/num_correct": float(correct),
            f"{tag}/total_reward": reward_total,
        }

        if total > 0:
            metrics[f"{tag}/accuracy"] = correct / total
            metrics[f"{tag}/avg_reward"] = reward_total / total

        accelerator.log(metrics, step=global_step)


# --- Main training loop -----------------------------------------------------


def main():
    config = get_config()

    project_name = config.experiment.project_name
    project_dir = config.experiment.project_dir

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = os.path.join(project_dir, "logs")

    config.training.gradient_accumulation_steps = int(
        config.training.max_generation_length * config.training.batch_size * config.training.group_size / 
        config.training.micro_batch_size / config.training.tokens_per_step
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=project_name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        accelerator.init_trackers(
            project_name,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(project_dir, exist_ok=True)
        config_path = os.path.join(project_dir, "config.yaml")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    pretrained_model = config.model.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    mask_token = config.training.mask_token
    config.training.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    config.training.pad_token_id = tokenizer.convert_tokens_to_ids(config.training.pad_token)

    model = LLaDAModelLM.from_pretrained(
        pretrained_model,
        torch_dtype=torch.bfloat16 if config.training.mixed_precision == "bf16" else None,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
        eps=config.optimizer.epsilon,
    )

    train_dataset = MathPromptDataset(config.dataset.path)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle,
        collate_fn=lambda batch: batch,
    )

    val_dataloader: Optional[DataLoader] = None
    if config.dataset.get("val_path", None):
        val_dataset = MathPromptDataset(config.dataset.val_path)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

    test_dataloader: Optional[DataLoader] = None
    if config.dataset.get("test_path", None):
        test_dataset = MathPromptDataset(config.dataset.test_path)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

    updates_per_rollout = max(int(config.training.get("updates_per_rollout", 1)), 1)
    train_batch_size = int(
        config.training.get("train_batch_size", config.training.batch_size)
    )
    config.training.train_batch_size = train_batch_size
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) * updates_per_rollout
        / max(int(config.training.gradient_accumulation_steps), 1)
    )
    max_train_steps = num_update_steps_per_epoch * int(config.training.num_train_epochs)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        num_training_steps=max_train_steps,
        min_lr_scale=config.lr_scheduler.min_lr_scale,
    )

    prepare_items = [model, optimizer, lr_scheduler, train_dataloader]
    dataloader_indices: Dict[str, int] = {}

    if val_dataloader is not None:
        dataloader_indices["val"] = len(prepare_items)
        prepare_items.append(val_dataloader)

    if test_dataloader is not None:
        dataloader_indices["test"] = len(prepare_items)
        prepare_items.append(test_dataloader)

    prepared = accelerator.prepare(*prepare_items)
    model, optimizer, lr_scheduler, train_dataloader = prepared[:4]

    if "val" in dataloader_indices:
        val_dataloader = prepared[dataloader_indices["val"]]
    if "test" in dataloader_indices:
        test_dataloader = prepared[dataloader_indices["test"]]

    # Resume training ----------------------------------------------------------
    resume_dir = config.experiment.get("resume_from_checkpoint", None)
    global_step = 0
    skip_batches = 0
    saved_batches_seen: Optional[int] = None

    if resume_dir:
        resume_path = Path(resume_dir)
        accelerator.print(f"Resuming training from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)

        sched_path = resume_path / "lr_scheduler.pt"
        if sched_path.exists():
            lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

        ckpt_name = resume_path.name
        if ckpt_name.startswith("step_"):
            parts = ckpt_name.split("_")
            if len(parts) > 1:
                try:
                    global_step = int(parts[1])
                except Exception:
                    global_step = 0

            for idx, part in enumerate(parts):
                if part == "batch" and idx + 1 < len(parts):
                    try:
                        saved_batches_seen = int(parts[idx + 1])
                    except Exception:
                        saved_batches_seen = None

        metadata_path = resume_path / "metadata.json"
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                if saved_batches_seen is None and "batches_seen" in meta:
                    saved_batches_seen = int(meta["batches_seen"])
                if global_step == 0:
                    step_val = meta.get("step")
                    if isinstance(step_val, str) and step_val.startswith("step_"):
                        try:
                            global_step = int(step_val.split("_")[1])
                        except Exception:
                            global_step = 0
            except Exception:
                pass

        skip_batches = (
            saved_batches_seen
            if saved_batches_seen is not None
            else global_step // updates_per_rollout
        )
        accelerator.print(
            f"Skipping the first {skip_batches} batches after resume (from checkpoint)."
        )
    else:
        accelerator.print("Starting training from scratch.")

    model.train()

    num_epochs = int(config.training.num_train_epochs)
    validation_interval = int(config.training.get("validation_interval", 0))
    checkpoint_interval = int(config.training.get("checkpoint_interval", 0))

    num_batches_per_epoch = len(train_dataloader)
    start_epoch = min(num_epochs - 1, skip_batches // max(num_batches_per_epoch, 1))
    skip_batches_in_epoch = skip_batches % max(num_batches_per_epoch, 1)
    train_on_sample_once = bool(config.training.get("train_on_sample_once", False))
    reuse_before_sample = bool(config.training.get("reuse_before_sample", True))
    global_batch_size = train_batch_size * accelerator.num_processes

    sample_queue: Deque[SampleTrajectoryBundle] = deque()
    recent_missing_signals: Deque[bool] = deque(maxlen=100)
    pending_stats = SamplingStats()
    use_prompt_cache = bool(config.training.get("enable_prompt_cache", True))
    prompt_cache = PromptCacheManager() if use_prompt_cache else None
    rollout_batches_seen = skip_batches

    for epoch in range(start_epoch, num_epochs):
        batch_iter = tqdm(
            iterable=enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch_idx, batch in batch_iter:
            if epoch == start_epoch and skip_batches_in_epoch > 0 and batch_idx < skip_batches_in_epoch:
                continue
            step_records, completed, stats = collect_rollouts(
                batch,
                model,
                tokenizer,
                config,
                accelerator,
                progress_desc=f"process: {accelerator.process_index} Rollout E{epoch+1}",
                enable_bar=config.training.debug,
                prompt_cache=prompt_cache,
            )
            if prompt_cache is not None:
                prompt_cache.clear()

            if bool(config.training.get("debug_rollout", False)):
                assign_advantages(
                    step_records,
                    completed,
                    tokenizer,
                    config.training.group_size,
                )
                _store_debug_rollouts(
                    completed,
                    step_records,
                    tokenizer,
                    config.training.get("debug_rollout_store_path", ""),
                    accelerator,
                )

            gathered_step_records = gather_object([step_records])
            gathered_completed = gather_object([completed])
            gathered_stats = gather_object([stats])
            gathered_counts = gather_object([len(batch)])
            merged_stats = SamplingStats()
            for proc_stats in gathered_stats:
                merged_stats.merge(proc_stats)
            pending_stats.merge(merged_stats)

            all_step_records: List[StepRecord] = []
            all_completed: List[CompletedTrajectory] = []

            sample_offset = 0
            for proc_idx, proc_steps in enumerate(gathered_step_records):
                proc_completed = gathered_completed[proc_idx]
                for step in proc_steps:
                    step.sample_idx += sample_offset
                for traj in proc_completed:
                    traj.sample_idx += sample_offset

                all_step_records.extend(proc_steps)
                all_completed.extend(proc_completed)
                sample_offset += gathered_counts[proc_idx]

            new_samples, missing_signal_flags, all_correct, all_error = _filter_sample_trajectories(all_step_records, all_completed, tokenizer, config.training.filter_invalid)
            _reindex_sample_trajectories(new_samples, 0) # re_index from zero
            sample_queue.extend(new_samples)
            recent_missing_signals.extend(missing_signal_flags)
            rollout_batches_seen += 1
            if accelerator.is_main_process and config.training.debug:
                print(f"total: {len(missing_signal_flags)} missing:{sum(missing_signal_flags)}")
                print(f"all_correct: {all_correct} all_error: {all_error}")

            while len(sample_queue) >= global_batch_size:
                batch_samples: List[SampleTrajectoryBundle] = [
                    sample_queue.popleft() for _ in range(global_batch_size)
                ]
                _reindex_sample_trajectories(batch_samples, 0)

                training_step_records = [
                    step for bundle in batch_samples for step in bundle.step_records
                ]
                training_completed = [
                    traj for bundle in batch_samples for traj in bundle.completed
                ]

                raw_reward_sum = assign_advantages(
                    training_step_records,
                    training_completed,
                    tokenizer,
                    config.training.group_size,
                )

                update_batches: List[Tuple[List[SampleTrajectoryBundle], int]] = []
                if train_on_sample_once:
                    per_device_base = train_batch_size // updates_per_rollout
                    remainder = train_batch_size % updates_per_rollout
                    cursor = 0

                    for update_idx in range(updates_per_rollout):
                        per_device_size = per_device_base + (
                            1 if update_idx < remainder else 0
                        )
                        if per_device_size <= 0:
                            continue

                        per_global_size = per_device_size * accelerator.num_processes
                        update_batch = batch_samples[cursor : cursor + per_global_size]
                        cursor += per_global_size
                        if not update_batch:
                            continue
                        update_batches.append((update_batch, per_device_size)) # （一个batch的所有device的数据，每个device取多少训）

                    if not update_batches:
                        update_batches.append((batch_samples, train_batch_size))
                else:
                    update_batches.append((batch_samples, train_batch_size)) # （所有数据，每个batch取多少训）

                stats_for_logging = pending_stats
                pending_stats = SamplingStats()

                for update_batch, per_device_size in update_batches:
                    local_start = accelerator.process_index * per_device_size
                    local_end = local_start + per_device_size
                    local_batch_samples = update_batch[local_start:local_end]

                    local_training_step_records = [
                        step for bundle in local_batch_samples for step in bundle.step_records
                    ]
                    loss_desc = (
                        f"process{accelerator.process_index} Loss E{epoch + 1}/{num_epochs} B{batch_idx + 1}/{len(train_dataloader)}"
                    )
                    rl_loss, clip_fraction, rl_steps, ratio_debug = compute_losses(
                        local_training_step_records,
                        model,
                        tokenizer,
                        config,
                        accelerator,
                        optimizer,
                        lr_scheduler,
                        progress_desc=loss_desc,
                        prompt_cache=prompt_cache,
                        collect_ratio_debug=bool(
                            config.training.get("debug_ratio", False)
                        ),
                        enable_bar=config.training.debug
                    )

                    if accelerator.sync_gradients:
                        if bool(config.training.get("debug_ratio", False)):
                            step_id = global_step
                            gathered_ratios = gather_object([ratio_debug])
                            if accelerator.is_main_process:
                                locate_entries: List[Any] = []
                                token_entries: List[Any] = []
                                trajectory_entries: List[Any] = []
                                for proc_ratios in gathered_ratios:
                                    if not proc_ratios:
                                        continue
                                    locate_entries.extend(proc_ratios.get("locate_ratio", []))
                                    token_entries.extend(proc_ratios.get("token_ratio", []))
                                    trajectory_entries.extend(
                                        proc_ratios.get("trajectory_ratio", [])
                                    )

                                locate_entries.sort(key=lambda x: x[0] if x else 0.0)
                                token_entries.sort(key=lambda x: x[0] if x else 0.0)
                                trajectory_entries.sort(
                                    key=lambda x: x.get("combined", (0.0,))[0]
                                    if isinstance(x, dict)
                                    else 0.0
                                )

                                debug_ratio_path = Path(project_dir) / config.training.debug_ratio_path
                                existing_debug: Dict[str, Any] = {}
                                if debug_ratio_path.exists():
                                    with open(debug_ratio_path, "r", encoding="utf-8") as fh:
                                        existing_debug = json.load(fh)

                                existing_debug[str(step_id)] = {
                                    "token_ratio": token_entries,
                                    "locate_ratio": locate_entries,
                                    "trajectory_ratio": trajectory_entries,
                                }

                                with open(debug_ratio_path, "w", encoding="utf-8") as fh:
                                    json.dump(existing_debug, fh, ensure_ascii=False, indent=2)

                        global_step += 1
                        metrics = {
                            "loss/rl": rl_loss,
                            "train/clip_fraction_tok": clip_fraction[0],
                            "train/clip_fraction_loc": clip_fraction[1],
                            "train/raw_reward_sum": raw_reward_sum,
                            "train/rl_steps": rl_steps,
                        }

                        if recent_missing_signals:
                            metrics["train/recent_missing_signals_100"] = float(
                                sum(recent_missing_signals)
                            )
                        if stats_for_logging.entropy_count > 0:
                            metrics["train/location_entropy"] = (
                                stats_for_logging.entropy_sum / stats_for_logging.entropy_count
                            )
                        if stats_for_logging.max_prob_count > 0:
                            metrics["train/location_max_prob"] = (
                                stats_for_logging.max_prob_sum / stats_for_logging.max_prob_count
                            )

                        accelerator.log(metrics, step=global_step)

                        if (
                            validation_interval > 0
                            and val_dataloader is not None
                            and global_step % validation_interval == 0
                        ):
                            run_evaluation(
                                accelerator,
                                model,
                                tokenizer,
                                config,
                                val_dataloader,
                                global_step,
                                "val",
                            )
                            model.train()

                        if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                            ckpt_label = f"step_{global_step:06d}_batch_{rollout_batches_seen:06d}"
                            ckpt_dir = (
                                Path(project_dir)
                                / "ckpt"
                                / config.model.optimized_name
                                / ckpt_label
                            )
                            ckpt_dir.mkdir(parents=True, exist_ok=True)
                            save_checkpoint(
                                model,
                                tokenizer,
                                accelerator,
                                config,
                                project_dir,
                                ckpt_label,
                                save_training_state=config.experiment.get("save_training_state", True),
                                batches_seen=rollout_batches_seen,
                            )
                            if accelerator.is_main_process:
                                torch.save(lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")
                if not reuse_before_sample:
                    sample_queue.clear()
                    break
    # Final evals & checkpoint
    run_evaluation(accelerator, model, tokenizer, config, val_dataloader, global_step, "val_final")
    run_evaluation(accelerator, model, tokenizer, config, test_dataloader, global_step, "test")

    final_ckpt = Path(project_dir) / "ckpt" / config.model.optimized_name / "final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        model,
        tokenizer,
        accelerator,
        config,
        project_dir,
        "final",
        save_training_state=config.experiment.get("save_training_state", True),
        batches_seen=rollout_batches_seen,
    )
    if accelerator.is_main_process:
        torch.save(lr_scheduler.state_dict(), final_ckpt / "lr_scheduler.pt")

    accelerator.end_training()


if __name__ == "__main__":
    main()
