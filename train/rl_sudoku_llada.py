import os
import sys
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf

import wandb
from transformers import AutoTokenizer

from models import LLaDAModelLM
from models.sampling import gumbel_sample
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_error, set_verbosity_info

from train.utils import get_config, flatten_omega_conf
from train.sudoku_tools import detect_definite, judge_error
from train.sudoku_rl_utils import (
    build_location_distribution,
    build_token_distribution,
    compute_f_theta_value,
    pad_to_length,
    prepare_sequence,
    compute_definites,
    save_checkpoint,
    run_validation,
    SudokuPromptDataset,
    StepRecord,
    SamplingStats,
    PreparedSample,
    TrajectoryJob,
)

logger = get_logger(__name__, log_level="INFO")

def collect_rollouts(
    batch: Sequence[Dict[str, Any]],
    model: LLaDAModelLM,
    tokenizer,
    config,
    device: torch.device,
) -> Tuple[List[StepRecord], SamplingStats]:
    training_cfg = config.training
    mask_token_id = training_cfg.mask_token_id
    stats = SamplingStats()

    prepared_samples: List[PreparedSample] = []
    for sample in batch:
        prepared_samples.append(
            prepare_sequence(
                sample,
                tokenizer,
                mask_token_id,
                training_cfg.max_generation_length,
            )
        )

    trajectories_per_sample: Dict[int, List[List[StepRecord]]] = {
        idx: [] for idx in range(len(prepared_samples))
    }

    rollout_batch_size = max(int(training_cfg.rollout_batch_size), 1)
    job_queue: Deque[TrajectoryJob] = deque()

    for sample_idx, prepared in enumerate(prepared_samples):
        mask_count = int((prepared.state == mask_token_id).sum().item())
        map_index_set = set(int(idx) for idx in prepared.map_indices.tolist())
        for group_idx in range(training_cfg.group_size):
            job_queue.append(
                TrajectoryJob(
                    sample_idx=sample_idx,
                    trajectory_idx=group_idx,
                    state=prepared.state.clone().to(device),
                    mask_start=prepared.mask_start,
                    map_indices=prepared.map_indices.clone(),
                    map_index_set=set(map_index_set),
                    max_steps=mask_count,
                    metadata=prepared.metadata,
                )
            )

    while job_queue:
        active_jobs: List[TrajectoryJob] = []
        while job_queue and len(active_jobs) < rollout_batch_size:
            job = job_queue.popleft()
            active_jobs.append(job)

        if not active_jobs:
            continue

        state_batch = torch.stack([job.state for job in active_jobs], dim=0).to(device)

        with torch.no_grad():
            logits_batch = model(state_batch).logits

        for batch_idx, job in enumerate(active_jobs):
            logits = logits_batch[batch_idx].to(torch.float32)
            candidate_mask = job.state.eq(mask_token_id)

            if (not torch.any(candidate_mask)) or job.steps_taken >= job.max_steps:
                job.terminated = True
                if job.records:
                    trajectories_per_sample[job.sample_idx].append(job.records)
                stats.finish_trajectory(error=False)
                continue

            loc_probs, loc_scores = build_location_distribution(
                logits,
                candidate_mask=candidate_mask,
                epsilon=training_cfg.epsilon_small,
                temperature=training_cfg.location_temperature,
            )

            valid_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                job.terminated = True
                if job.records:
                    trajectories_per_sample[job.sample_idx].append(job.records)
                stats.finish_trajectory(error=False)
                continue

            valid_probs = loc_probs[valid_indices]
            entropy = -(valid_probs * torch.log(valid_probs + 1e-12)).sum()
            stats.update_entropy(float(entropy))
            stats.update_max_prob(float(valid_probs.max()))

            valid_scores = loc_scores[valid_indices]
            greedy_idx_rel = torch.argmax(valid_scores)
            greedy_abs_idx = int(valid_indices[greedy_idx_rel].item())

            definites_indices = set(
                compute_definites(job.state, job.mask_start, tokenizer)
            )
            stats.update_greedy(
                in_map=greedy_abs_idx in job.map_index_set,
                in_s1=greedy_abs_idx in definites_indices,
            )

            loc_logits = torch.where(
                candidate_mask,
                loc_scores,
                torch.full_like(loc_scores, -1e9),
            )
            location_index = int(gumbel_sample(loc_logits).item())
            logprob_loc = float(torch.log(loc_probs[location_index] + 1e-12))

            token_logits_scaled = logits[location_index] / max(
                training_cfg.token_temperature, 1e-8
            )
            token_id = int(gumbel_sample(token_logits_scaled).item())
            token_log_probs = torch.log_softmax(token_logits_scaled, dim=-1)
            logprob_tok = float(token_log_probs[token_id])

            next_state = job.state.clone()
            next_state[location_index] = token_id

            is_error = bool(judge_error(next_state.detach().cpu().tolist(), job.mask_start))

            step_record = StepRecord(
                sample_idx=job.sample_idx,
                trajectory_idx=job.trajectory_idx,
                step_index=job.steps_taken,
                state_before=job.state.detach().cpu(),
                state_after=next_state.detach().cpu(),
                location_index=location_index,
                token_id=token_id,
                logprob_old_loc=logprob_loc,
                logprob_old_tok=logprob_tok,
                mask_start=job.mask_start,
                map_indices=job.map_indices.clone(),
                metadata=job.metadata,
            )

            stats.record_step(0.0)

            if is_error:
                job.records.append(step_record)
                job.terminated = True
                if job.records:
                    trajectories_per_sample[job.sample_idx].append(job.records)
                stats.finish_trajectory(error=True)
                continue

            job.state = next_state
            job.steps_taken += 1
            job.records.append(step_record)
            job_queue.append(job)

    flattened: List[StepRecord] = []
    for trajs in trajectories_per_sample.values():
        for traj in trajs:
            flattened.extend(traj)

    return flattened, stats


def compute_losses(
    step_records: List[StepRecord],
    model: LLaDAModelLM,
    tokenizer,
    config,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    if not step_records:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, 0.0, 0.0

    training_cfg = config.training
    mask_token_id = training_cfg.mask_token_id
    rollout_batch_size = max(int(training_cfg.rollout_batch_size), 1)

    ratio_records: List[Tuple[torch.Tensor, torch.Tensor, StepRecord]] = []
    sft_losses: List[torch.Tensor] = []
    clip_events = 0

    for start in range(0, len(step_records), rollout_batch_size):
        batch_records = step_records[start : start + rollout_batch_size]
        states_before = torch.stack([step.state_before for step in batch_records]).to(device)

        logits_before = model(states_before).logits

        for idx, step in enumerate(batch_records):
            logits_b = logits_before[idx]
            state_before = states_before[idx]
            candidate_mask = state_before.eq(mask_token_id)
            if not torch.any(candidate_mask):
                step.f_theta_value = 0.0
                step.advantage = 0.0
                continue

            loc_probs, _ = build_location_distribution(
                logits_b,
                candidate_mask=candidate_mask,
                epsilon=training_cfg.epsilon_small,
                temperature=training_cfg.location_temperature,
            )
            loc_prob_selected = loc_probs[step.location_index]
            logprob_new_loc = torch.log(loc_prob_selected + 1e-12)

            _, token_log_probs = build_token_distribution(
                logits_b[step.location_index],
                temperature=training_cfg.token_temperature,
            )
            logprob_new_tok = token_log_probs[step.token_id]

            logprob_old = torch.tensor(step.old_logprob_sum, device=device, dtype=logits_b.dtype)
            logprob_new = logprob_new_loc + logprob_new_tok
            ratio = torch.exp(logprob_new - logprob_old)

            clip_ratio = torch.clamp(ratio, 1 - training_cfg.clip_epsilon, 1 + training_cfg.clip_epsilon)
            if clip_ratio.item() != ratio.item():
                clip_events += 1

            ratio_records.append((ratio, clip_ratio, step))

            if step.is_error:
                step.f_theta_value = 0.0
                continue

            f_theta_val = compute_f_theta_value(
                logits_b,
                states_before[idx],
                mask_token_id=mask_token_id,
                mask_start=step.mask_start,
                tokenizer=tokenizer,
                lambda_coeff=training_cfg.lambda_coeff,
                epsilon=training_cfg.epsilon_small,
                location_temperature=training_cfg.location_temperature,
                token_temperature=training_cfg.token_temperature,
                map_indices=step.map_indices.to(device) if step.map_indices is not None else None,
                detect_fn=detect_definite,
            )
            step.f_theta_value = float(f_theta_val.detach().cpu())
            sft_losses.append(-ratio.detach() * f_theta_val)

    trajectories: Dict[int, Dict[int, List[StepRecord]]] = {}
    for step in step_records:
        sample_map = trajectories.setdefault(step.sample_idx, {})
        sample_map.setdefault(step.trajectory_idx, []).append(step)

    gamma = training_cfg.gamma
    for sample_trajs in trajectories.values():
        traj_entries: List[Tuple[int, List[StepRecord]]] = []
        for traj_idx, steps in sample_trajs.items():
            sorted_steps = sorted(steps, key=lambda item: item.step_index)
            traj_entries.append((traj_idx, sorted_steps))

        if not traj_entries:
            continue

        max_steps = max(len(steps) for _, steps in traj_entries)
        if max_steps == 0:
            continue

        reward_rows: List[torch.Tensor] = []
        for _, steps in traj_entries:
            rewards = [float(step.f_theta_value) for step in steps]
            reward_rows.append(pad_to_length(rewards, max_steps))

        rewards_tensor = torch.stack(reward_rows, dim=0).to(device)
        discount_powers = torch.tensor(
            [gamma ** i for i in range(max_steps)], dtype=torch.float32, device=device
        )
        returns = torch.zeros_like(rewards_tensor)
        for t in range(max_steps):
            if t > 0:
                discounts = discount_powers[: -t].flip(0)
            else: discounts = discount_powers.flip(0)
            returns[:, t] = (rewards_tensor[:, t:] * discounts).sum(dim=-1)

        mean = returns.mean(dim=0)
        std = returns.std(dim=0, unbiased=False)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        advantages = (returns - mean) / std

        for row_idx, (_, steps) in enumerate(traj_entries):
            for col_idx, step in enumerate(steps):
                step.advantage = float(advantages[row_idx, col_idx].detach().cpu().item())

    rl_losses: List[torch.Tensor] = []
    for ratio, clip_ratio, step in ratio_records:
        adv = torch.tensor(step.advantage, device=device, dtype=ratio.dtype)
        rl_loss = -torch.min(ratio * adv, clip_ratio * adv)
        rl_losses.append(rl_loss)

    rl_loss = torch.stack(rl_losses).mean() if rl_losses else torch.tensor(0.0, device=device)
    sft_loss = torch.stack(sft_losses).mean() if sft_losses else torch.tensor(0.0, device=device)
    clip_fraction = clip_events / max(len(rl_losses), 1)

    reward_total = float(sum(step.f_theta_value for step in step_records))

    return rl_loss, sft_loss, clip_fraction, reward_total


def main():
    config = get_config()

    project_name = config.experiment.project
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = os.path.join(project_name, "logs")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=None,
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
            name=config.experiment.project,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.project, exist_ok=True)
        config_path = os.path.join(config.experiment.project, "config.yaml")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    pretrained_model = config.model.pretrained_model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    mask_token = config.training.mask_token
    config.training.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    config.training.pad_token_id = tokenizer.convert_tokens_to_ids(config.training.pad_token)

    model = LLaDAModelLM.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)

    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        weight_decay=config.optimizer.weight_decay,
        eps=config.optimizer.epsilon,
    )

    dataset = SudokuPromptDataset(config.dataset.path)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=lambda batch: batch,
    )

    val_dataloader: Optional[DataLoader] = None
    if getattr(config.dataset, "val_path", None):
        val_dataset = SudokuPromptDataset(config.dataset.val_path)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

    if val_dataloader is not None:
        model, optimizer, dataloader, val_dataloader = accelerator.prepare(
            model,
            optimizer,
            dataloader,
            val_dataloader,
        )
    else:
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / config.training.gradient_accumulation_steps)
    max_train_steps = num_update_steps_per_epoch * config.training.num_train_epochs

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        min_lr_scale=config.lr_scheduler.min_lr_scale,
    )

    global_step = 0
    validation_interval = int(config.training.get("validation_interval", 0) or 0)
    checkpoint_interval = int(config.training.get("checkpoint_interval", 0) or 0)

    for _ in range(config.training.num_train_epochs):
        accumulated_stats = SamplingStats()
        for batch in dataloader:
            with accelerator.accumulate(model):
                step_records: List[StepRecord] = []
                batch_stats = SamplingStats()

                model.eval()
                records, rollout_stats = collect_rollouts(
                    batch,
                    model,
                    tokenizer,
                    config,
                    accelerator.device,
                )
                step_records.extend(records)
                batch_stats.merge(rollout_stats)
                model.train()

                accumulated_stats.merge(batch_stats)
                rl_loss, sft_loss, clip_fraction, reward_total = compute_losses(
                    step_records,
                    model,
                    tokenizer,
                    config,
                    accelerator.device,
                )

                accumulated_stats.reward_sum += reward_total

                total_loss = rl_loss + config.training.sft_loss_scale * sft_loss

                accelerator.backward(total_loss)
                if accelerator.sync_gradients and config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                if accelerator.sync_gradients:
                    global_step += 1

                    metrics = {
                        "loss/total": total_loss.detach().item(),
                        "loss/rl": rl_loss.detach().item(),
                        "loss/sft": sft_loss.detach().item(),
                        "train/clip_fraction": clip_fraction,
                        "train/rollout_steps": accumulated_stats.step_count,
                        "train/trajectories": accumulated_stats.trajectory_count,
                        "train/error_trajectories": accumulated_stats.error_terminated,
                        "train/reward_sum": accumulated_stats.reward_sum,
                    }

                    if accumulated_stats.entropy_count > 0:
                        metrics["train/location_entropy"] = accumulated_stats.entropy_sum / accumulated_stats.entropy_count
                    if accumulated_stats.max_prob_count > 0:
                        metrics["train/location_max_prob"] = accumulated_stats.max_prob_sum / accumulated_stats.max_prob_count
                    if accumulated_stats.greedy_inside_map > 0:
                        if accumulated_stats.step_count > 0:
                            metrics["train/greedy_in_map_ratio"] = accumulated_stats.greedy_inside_map / accumulated_stats.step_count
                        metrics["train/greedy_s1_ratio"] = accumulated_stats.greedy_inside_s1 / accumulated_stats.greedy_inside_map
                    if accumulated_stats.step_count > 0:
                        metrics["train/avg_reward"] = accumulated_stats.reward_sum / accumulated_stats.step_count
                    if accumulated_stats.trajectory_count > 0:
                        metrics["train/error_traj_fraction"] = accumulated_stats.error_terminated / accumulated_stats.trajectory_count
                        metrics["train/avg_traj_len"] = accumulated_stats.step_count / accumulated_stats.trajectory_count

                    accelerator.log(metrics, step=global_step)
                    accumulated_stats = SamplingStats()

                    if (
                        validation_interval > 0
                        and val_dataloader is not None
                        and global_step % validation_interval == 0
                    ):
                        run_validation(accelerator, model, tokenizer, config, val_dataloader, global_step)
                        model.train()

                    if checkpoint_interval > 0 and global_step % checkpoint_interval == 0:
                        save_checkpoint(model, tokenizer, accelerator, config, project_name, global_step)

        accelerator.wait_for_everyone()

    save_checkpoint(model, tokenizer, accelerator, config, project_name, "final")

    accelerator.end_training()


if __name__ == "__main__":
    main()