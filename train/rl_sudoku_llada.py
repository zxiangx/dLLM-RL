import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
import math

import torch
from torch.distributions import Categorical
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf

import wandb
from transformers import AutoTokenizer

from models import LLaDAModelLM
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_error, set_verbosity_info

from train.utils import get_config, flatten_omega_conf
from train.sudoku_tools import detect_definite, judge_error, pre_fill
from train.sudoku_rl_utils import (
    build_location_distribution,
    build_token_distribution,
    compute_f_theta_value,
    pad_to_length,
)


logger = get_logger(__name__, log_level="INFO")


class SudokuPromptDataset(Dataset):
    """Dataset that exposes raw Sudoku prompts.

    The dataset file must contain a JSON list or JSONL file where each item is a
    mapping with at least a ``"prompt"`` field.  Optional fields:

    ``mask_ids``
        Token id sequence representing the masked Sudoku instance.  When omitted
        the prompt is tokenised directly which assumes that the prompt already
        contains mask tokens.

    ``metadata``
        Free form dictionary that is forwarded to the Sudoku helper functions.
    """

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
            with open(data_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    # allow {"data": [...]} style payloads
                    if "data" in data:
                        data = data["data"]
                    else:
                        raise ValueError(
                            "JSON dataset must be a list of samples or contain a "
                            "'data' field."
                        )
                records.extend(data)

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
class StepRecord:
    state_before: torch.Tensor
    state_after: torch.Tensor
    location_index: int
    token_id: int
    logprob_old_loc: float
    logprob_old_tok: float
    reward_old: float
    mask_start: int
    map_indices: Optional[torch.Tensor]
    metadata: Dict[str, Any] = field(default_factory=dict)
    advantage: float = 0.0

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


def _prepare_sequence( # 需要大改
    sample: Dict[str, Any],
    tokenizer,
    mask_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    prompt = sample["prompt"]

    if "mask_ids" in sample and sample["mask_ids"] is not None:
        base_ids = torch.tensor(sample["mask_ids"], dtype=torch.long)
    else:
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        base_ids = encoded["input_ids"].squeeze(0)

    mask_positions = (base_ids == mask_token_id).nonzero(as_tuple=False).flatten()
    if len(mask_positions) == 0:
        raise ValueError("Mask token never appears in the provided sequence")
    mask_start = int(mask_positions.min().item())

    filled_ids, filled_count, filled_indices = pre_fill(prompt, base_ids.tolist())
    filled_tensor = torch.tensor(filled_ids, dtype=torch.long)
    filled_indices_abs = torch.tensor(
        [mask_start + int(idx) for idx in filled_indices], dtype=torch.long
    )

    map_indices_abs: Optional[torch.Tensor]
    if "map_indices" in sample and sample["map_indices"] is not None:
        map_indices_abs = torch.tensor(
            [mask_start + int(idx) for idx in sample["map_indices"]],
            dtype=torch.long,
        )
    else:
        map_indices_abs = mask_positions.to(dtype=torch.long)

    return filled_tensor, map_indices_abs, mask_start, filled_indices_abs


def _compute_definites(
    state_ids: torch.Tensor,
    mask_start: int,
    tokenizer,
) -> List[int]:
    from train.sudoku_rl_utils import normalise_definite_positions

    definites_raw = detect_definite(state_ids.tolist())
    definites = normalise_definite_positions(definites_raw, mask_start, tokenizer)
    return [pos.index for pos in definites]


def _sample_single_trajectory(
    model: LLaDAModelLM,
    tokenizer,
    config,
    device: torch.device,
    base_state: torch.Tensor,
    map_indices: torch.Tensor,
    mask_start: int,
    filled_indices_abs: torch.Tensor,
    stats: SamplingStats,
) -> List[StepRecord]:
    training_cfg = config.training
    mask_token_id = training_cfg.mask_token_id

    remaining_map_indices = {int(idx) for idx in map_indices.tolist()}
    prefilled_abs = {int(idx) for idx in filled_indices_abs.tolist()}
    steps_budget = max(len(remaining_map_indices - prefilled_abs), 0)

    trajectory: List[StepRecord] = []

    state = base_state.clone()

    for _ in range(steps_budget):
        candidate_mask = state.eq(mask_token_id)
        if not torch.any(candidate_mask):
            break

        with torch.no_grad():
            logits = model(state.unsqueeze(0).to(device)).logits[0]

        loc_probs, loc_logits = build_location_distribution(
            logits,
            candidate_mask=candidate_mask.to(device),
            epsilon=training_cfg.epsilon_small,
            temperature=training_cfg.location_temperature,
        )

        loc_probs = loc_probs.cpu()
        loc_logits = loc_logits.cpu()

        dist = Categorical(probs=loc_probs + 1e-12)
        stats.update_entropy(dist.entropy().item())
        stats.update_max_prob(loc_probs.max().item())

        greedy_index = int(torch.argmax(loc_logits).item())
        definites_indices = set(
            _compute_definites(state, mask_start, tokenizer)
        )
        stats.update_greedy(
            in_map=greedy_index in set(map_indices.tolist()),
            in_s1=greedy_index in definites_indices,
        )

        location_index = int(dist.sample().item())
        loc_logprob = float(torch.log(loc_probs[location_index] + 1e-12))

        token_probs, token_log_probs = build_token_distribution(
            logits[location_index],
            temperature=training_cfg.token_temperature,
        )
        token_probs = token_probs.cpu()
        token_log_probs = token_log_probs.cpu()

        token_dist = Categorical(probs=token_probs + 1e-12)
        token_id = int(token_dist.sample().item())
        tok_logprob = float(token_log_probs[token_id])

        next_state = state.clone()
        next_state[location_index] = token_id

        is_error = judge_error(next_state.tolist())

        if is_error:
            reward_val = 0.0
        else:
            with torch.no_grad():
                logits_after = model(next_state.unsqueeze(0).to(device)).logits[0]
                reward_tensor = compute_f_theta_value(
                    logits_after,
                    next_state.to(device),
                    mask_token_id=mask_token_id,
                    mask_start=mask_start,
                    tokenizer=tokenizer,
                    lambda_coeff=training_cfg.lambda_coeff,
                    epsilon=training_cfg.epsilon_small,
                    location_temperature=training_cfg.location_temperature,
                    token_temperature=training_cfg.token_temperature,
                    map_indices=map_indices.to(device),
                    detect_fn=detect_definite,
                )
            reward_val = float(reward_tensor.detach().cpu())

        trajectory.append(
            StepRecord(
                state_before=state.clone(),
                state_after=next_state.clone(),
                location_index=location_index,
                token_id=token_id,
                logprob_old_loc=loc_logprob,
                logprob_old_tok=tok_logprob,
                reward_old=reward_val,
                mask_start=mask_start,
                map_indices=map_indices.clone(),
                metadata={},
            )
        )

        state = next_state

        if is_error:
            break

    return trajectory


def _assign_advantages(
    trajectories: List[List[StepRecord]],
    gamma: float,
) -> None:
    if not trajectories:
        return

    max_steps = max(len(traj) for traj in trajectories)
    if max_steps == 0:
        return

    rewards = torch.stack([
        pad_to_length([step.reward_old for step in traj], max_steps)
        for traj in trajectories
    ])

    discount_powers = torch.tensor(
        [gamma ** i for i in range(max_steps)], dtype=torch.float32
    )

    returns = torch.zeros_like(rewards)
    for t in range(max_steps):
        discounts = discount_powers[:-t].flip(0)
        returns[:, t] = (rewards[:, t:] * discounts).sum(dim=-1)

    mean = returns.mean(dim=0)
    std = returns.std(dim=0, unbiased=False)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    advantages = (returns - mean) / std

    for traj_idx, traj in enumerate(trajectories):
        for step_idx, step in enumerate(traj):
            step.advantage = float(advantages[traj_idx, step_idx].item())


def sample_trajectories_for_sample(
    sample: Dict[str, Any],
    model: LLaDAModelLM,
    tokenizer,
    config,
    device: torch.device,
) -> Tuple[List[StepRecord], SamplingStats]:
    training_cfg = config.training
    stats = SamplingStats()

    base_state, map_indices, mask_start, filled_indices_abs = _prepare_sequence(
        sample, tokenizer, training_cfg.mask_token_id
    )

    trajectories: List[List[StepRecord]] = []
    for _ in range(training_cfg.group_size):
        traj = _sample_single_trajectory(
            model,
            tokenizer,
            config,
            device,
            base_state,
            map_indices,
            mask_start,
            filled_indices_abs,
            stats,
        )
        if traj:
            trajectories.append(traj)

    _assign_advantages(trajectories, training_cfg.gamma)

    flattened: List[StepRecord] = []
    for traj in trajectories:
        flattened.extend(traj)

    return flattened, stats


def compute_losses(
    step_records: List[StepRecord],
    model: LLaDAModelLM,
    tokenizer,
    config,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if not step_records:
        zero = torch.tensor(0.0, device=device)
        return zero, zero, 0.0

    training_cfg = config.training
    mask_token_id = training_cfg.mask_token_id

    states_before = torch.stack([step.state_before for step in step_records]).to(device)
    states_after = torch.stack([step.state_after for step in step_records]).to(device)

    logits_before = model(states_before).logits
    logits_after = model(states_after).logits

    rl_losses = []
    sft_losses = []
    clip_events = 0

    for idx, step in enumerate(step_records):
        logits_b = logits_before[idx]
        candidate_mask = states_before[idx].eq(mask_token_id)
        if not torch.any(candidate_mask):
            continue

        loc_probs, _ = build_location_distribution(
            logits_b,
            candidate_mask=candidate_mask,
            epsilon=training_cfg.epsilon_small,
            temperature=training_cfg.location_temperature,
        )
        loc_prob_selected = loc_probs[step.location_index]
        logprob_new_loc = torch.log(loc_prob_selected + 1e-12)

        token_probs, token_log_probs = build_token_distribution(
            logits_b[step.location_index],
            temperature=training_cfg.token_temperature,
        )
        logprob_new_tok = token_log_probs[step.token_id]

        logprob_old = torch.tensor(step.old_logprob_sum, device=device)
        logprob_new = logprob_new_loc + logprob_new_tok
        ratio = torch.exp(logprob_new - logprob_old)

        clip_ratio = torch.clamp(ratio, 1 - training_cfg.clip_epsilon, 1 + training_cfg.clip_epsilon)
        if clip_ratio.item() != ratio.item():
            clip_events += 1

        adv = torch.tensor(step.advantage, device=device, dtype=logits_before.dtype)
        rl_loss = -torch.min(ratio * adv, clip_ratio * adv)
        rl_losses.append(rl_loss)

        logits_a = logits_after[idx]
        f_theta_val = compute_f_theta_value(
            logits_a,
            states_after[idx],
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
        sft_losses.append(-ratio.detach() * f_theta_val)

    rl_loss = torch.stack(rl_losses).mean() if rl_losses else torch.tensor(0.0, device=device)
    sft_loss = torch.stack(sft_losses).mean() if sft_losses else torch.tensor(0.0, device=device)
    clip_fraction = clip_events / max(len(rl_losses), 1)

    return rl_loss, sft_loss, clip_fraction


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
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
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

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    total_batch_size = (
        config.training.batch_size
        * accelerator.num_processes
        * config.training.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = math.ceil(len(dataloader) / total_batch_size)
    max_train_steps = num_update_steps_per_epoch * config.training.num_train_epochs

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.warmup_steps,
        min_lr_scale=config.lr_scheduler.min_lr_scale,
    )

    global_step = 0

    for epoch in range(config.training.num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                step_records: List[StepRecord] = []
                batch_stats = SamplingStats()

                model.eval()
                for sample in batch:
                    records, stats = sample_trajectories_for_sample(
                        sample,
                        model,
                        tokenizer,
                        config,
                        accelerator.device,
                    )
                    step_records.extend(records)
                    batch_stats.entropy_sum += stats.entropy_sum
                    batch_stats.entropy_count += stats.entropy_count
                    batch_stats.max_prob_sum += stats.max_prob_sum
                    batch_stats.max_prob_count += stats.max_prob_count
                    batch_stats.greedy_inside_map += stats.greedy_inside_map
                    batch_stats.greedy_inside_s1 += stats.greedy_inside_s1

                model.train()
                rl_loss, sft_loss, clip_fraction = compute_losses(
                    step_records,
                    model,
                    tokenizer,
                    config,
                    accelerator.device,
                )

                total_loss = rl_loss + config.training.sft_loss_scale * sft_loss

                accelerator.backward(total_loss)
                if config.training.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                metrics = {
                    "loss/total": total_loss.detach().item(),
                    "loss/rl": rl_loss.detach().item(),
                    "loss/sft": sft_loss.detach().item(),
                    "train/clip_fraction": clip_fraction,
                }

                if batch_stats.entropy_count > 0:
                    metrics["train/location_entropy"] = batch_stats.entropy_sum / batch_stats.entropy_count
                if batch_stats.max_prob_count > 0:
                    metrics["train/location_max_prob"] = batch_stats.max_prob_sum / batch_stats.max_prob_count
                if batch_stats.greedy_inside_map > 0:
                    metrics["train/greedy_s1_ratio"] = batch_stats.greedy_inside_s1 / batch_stats.greedy_inside_map

                accelerator.log(metrics, step=global_step)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        output_dir = os.path.join(project_name, "ckpt", config.model.optimized_name)
        accelerator.save_state(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
