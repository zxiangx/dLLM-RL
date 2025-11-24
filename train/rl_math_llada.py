import os
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from accelerate.utils import tqdm
from omegaconf import OmegaConf

import wandb
from transformers import AutoTokenizer
from math_verify import parse as math_parse, verify as math_verify

from models import LLaDAModelLM
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_error, set_verbosity_info
from models.sampling import gumbel_sample

from train.utils import flatten_omega_conf, get_config
from train.sudoku_rl_utils import (
    build_location_distribution,
    build_token_distribution,
    compute_logits_with_padding,
    save_checkpoint,
)
from sample.llada_sample import extract_final_boxed_answer

logger = get_logger(__name__, log_level="INFO")


# --- Data helpers -----------------------------------------------------------


class MathPromptDataset(Dataset):
    """Simple JSONL dataset for math prompts.

    Each record must contain a ``prompt`` field and is expected to provide a
    ground-truth ``answer`` used for reward computation.  Optional metadata
    fields are preserved verbatim.
    """

    def __init__(self, data_path: str):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not locate dataset: {data_path}")

        import json

        records: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:  # pragma: no cover - guard rail
                    raise ValueError(
                        "Dataset lines must be JSON objects containing at least a 'prompt' field"
                    ) from exc

        for record in records:
            if "prompt" not in record:
                raise ValueError("Each dataset entry must include a 'prompt' field")
            record.setdefault("answer", "")
            record.setdefault("metadata", {})

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


@dataclass
class PreparedSample:
    state: torch.Tensor
    prompt_length: int
    metadata: Dict[str, Any]
    answer: str


@dataclass
class StepRecord:
    sample_idx: int
    trajectory_idx: int
    step_index: int
    state_before: torch.Tensor
    location_index: int
    token_id: int
    logprob_old_loc: float
    logprob_old_tok: float
    prompt_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    advantage: float = 0.0

    @property
    def old_logprob_sum(self) -> float:
        return self.logprob_old_loc + self.logprob_old_tok


@dataclass
class TrajectoryJob:
    sample_idx: int
    trajectory_idx: int
    state: torch.Tensor
    prompt_length: int
    max_steps: int
    metadata: Dict[str, Any]
    answer: str
    records: List[StepRecord] = field(default_factory=list)
    steps_taken: int = 0


@dataclass
class CompletedTrajectory:
    sample_idx: int
    trajectory_idx: int
    final_state: torch.Tensor
    prompt_length: int
    metadata: Dict[str, Any]
    answer: str


@dataclass
class SamplingStats:
    entropy_sum: float = 0.0
    entropy_count: int = 0
    max_prob_sum: float = 0.0
    max_prob_count: int = 0

    def update_entropy(self, value: float) -> None:
        self.entropy_sum += float(value)
        self.entropy_count += 1

    def update_max_prob(self, value: float) -> None:
        self.max_prob_sum += float(value)
        self.max_prob_count += 1

    def merge(self, other: "SamplingStats") -> None:
        self.entropy_sum += other.entropy_sum
        self.entropy_count += other.entropy_count
        self.max_prob_sum += other.max_prob_sum
        self.max_prob_count += other.max_prob_count


@dataclass
class SampleTrajectoryBundle:
    step_records: List[StepRecord]
    completed: List[CompletedTrajectory]


# --- Core helpers -----------------------------------------------------------


def prepare_sequence(sample: Dict[str, Any], tokenizer, mask_token_id: int, max_generation_length: int) -> PreparedSample:
    prompt = sample["prompt"]
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer(chat_prompt, add_special_tokens=False)["input_ids"]

    filled_ids = list(prompt_ids)
    filled_ids.extend([mask_token_id] * max_generation_length)

    metadata = sample.get("metadata", {})
    return PreparedSample(
        state=torch.tensor(filled_ids, dtype=torch.long),
        prompt_length=len(prompt_ids),
        metadata=metadata,
        answer=str(sample.get("answer", "")),
    )


def collect_rollouts(
    batch: Sequence[Dict[str, Any]],
    model: LLaDAModelLM,
    tokenizer,
    config,
    accelerator: Accelerator,
    progress_desc: str = "",
    enable_bar: bool = False,
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
            location_index = int(gumbel_sample(loc_logits).item())
            logprob_loc = float(torch.log(loc_probs[location_index] + 1e-12))

            token_logits_scaled = logits[location_index] / max(training_cfg.token_temperature, 1e-8)
            token_id = int(gumbel_sample(token_logits_scaled).item())
            token_log_probs = torch.log_softmax(token_logits_scaled, dim=-1)
            logprob_tok = float(token_log_probs[token_id])

            next_state = job.state.clone()
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
                steps_finished += 1
                rollout_pbar.update(1)

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
    return tokenizer.decode(answer_tokens, skip_special_tokens=False).strip()


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
    step_records: List[StepRecord], completed: List[CompletedTrajectory], tokenizer
) -> List[SampleTrajectoryBundle]:
    per_sample_steps: Dict[int, List[StepRecord]] = defaultdict(list)
    per_sample_completed: Dict[int, List[CompletedTrajectory]] = defaultdict(list)

    for step in step_records:
        per_sample_steps[step.sample_idx].append(step)

    for traj in completed:
        per_sample_completed[traj.sample_idx].append(traj)

    filtered: List[SampleTrajectoryBundle] = []
    for sample_idx, trajs in per_sample_completed.items():
        steps = per_sample_steps.get(sample_idx, [])
        if not trajs or not steps:
            continue

        rewards = [
            _math_reward(_decode_answer(tokenizer, traj.final_state, traj.prompt_length), traj.answer)
            for traj in trajs
        ]
        has_correct = any(reward >= 2.0 for reward in rewards)
        has_incorrect = any(reward < 2.0 for reward in rewards)

        if not (has_correct and has_incorrect):
            continue

        filtered.append(SampleTrajectoryBundle(step_records=steps, completed=trajs))

    return filtered


def _reindex_sample_trajectories(
    trajectories: List[SampleTrajectoryBundle], start_index: int = 0
) -> int:
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
    progress_desc: str = "",
    enable_bar: bool = False,
) -> Tuple[float, float, float]:
    device = accelerator.device
    if not step_records:
        return 0.0, 0.0, 0.0

    training_cfg = config.training
    mask_token_id = training_cfg.mask_token_id
    micro_batch_size = max(int(training_cfg.micro_batch_size), 1)

    rl_loss_sum = torch.tensor(0.0, device=device)
    rl_count = 0
    clip_events = 0

    num_forward_batches = math.ceil(len(step_records) / micro_batch_size)
    loss_pbar = None
    if num_forward_batches > 0 and enable_bar:
        desc = progress_desc or "ComputeLoss"
        loss_pbar = tqdm(
            total=num_forward_batches,
            desc=desc,
            leave=False,
            disable=False,
        )

    for start in range(0, len(step_records), micro_batch_size):
        batch_records = step_records[start : start + micro_batch_size]
        states_before = [step.state_before.to(device) for step in batch_records]

        logits_before = compute_logits_with_padding(
            states_before,
            model,
            tokenizer,
            accelerator.device,
        )

        if loss_pbar is not None:
            loss_pbar.update(1)

        combined_losses = []
        for idx, step in enumerate(batch_records):
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
            loc_prob_selected = loc_probs[step.location_index]
            logprob_new_loc = torch.log(loc_prob_selected + 1e-12)

            _, token_log_probs = build_token_distribution(
                logits_b[step.location_index],
                temperature=training_cfg.token_temperature,
            )
            logprob_new_tok = token_log_probs[step.token_id]

            logprob_old = torch.tensor(step.old_logprob_sum, device=device, dtype=logprob_new_loc.dtype)
            logprob_new = logprob_new_loc + logprob_new_tok
            ratio = torch.exp(logprob_new - logprob_old)

            clip_ratio = torch.clamp(
                ratio, 1 - training_cfg.clip_epsilon, 1 + training_cfg.clip_epsilon
            )

            if (clip_ratio - ratio).abs() > 1e-8:
                clip_events += 1

            adv = torch.tensor(step.advantage, device=device, dtype=ratio.dtype)
            rl_term = -torch.min(ratio * adv, clip_ratio * adv)
            rl_loss_sum = rl_loss_sum + rl_term.detach()
            rl_count += 1

            combined_losses.append(rl_term / max(1, len(step_records)))

        total_loss = 0
        for loss in combined_losses:
            total_loss += loss
        accelerator.backward(total_loss)

    if loss_pbar is not None:
        loss_pbar.close()

    rl_loss_mean = float((rl_loss_sum / max(rl_count, 1)).item())
    clip_fraction = clip_events / max(rl_count, 1)

    return rl_loss_mean, clip_fraction, float(rl_count)


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
        total_samples += 1
        best_correct = 0.0
        for traj in trajs:
            output_text = _decode_answer(tokenizer, traj.final_state, traj.prompt_length)
            reward = _math_reward(output_text, traj.answer)
            reward_sum += reward
            best_correct = max(best_correct, float(_is_answer_correct(output_text, traj.answer)))

        if best_correct > 0:
            correct_samples += 1

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

    with torch.no_grad():
        eval_pbar = tqdm(
            total=len(dataloader),
            desc=f"{tag.title()}",
            disable=not accelerator.is_local_main_process,
            leave=False,
        )

        for batch in dataloader:
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

                    loc_logits = torch.where(
                        candidate_mask, loc_scores, torch.full_like(loc_scores, -1e9)
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

                    job_queue.append(job)

            batch_total, batch_correct, batch_reward = _decode_and_score(
                completed, tokenizer
            )

            total_samples += batch_total
            correct_samples += batch_correct
            reward_sum += batch_reward

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
        shuffle=True,
    )

    val_dataloader: Optional[DataLoader] = None
    if config.dataset.get("val_path", None):
        val_dataset = MathPromptDataset(config.dataset.val_path)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
        )

    test_dataloader: Optional[DataLoader] = None
    if config.dataset.get("test_path", None):
        test_dataset = MathPromptDataset(config.dataset.test_path)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
        )

    updates_per_rollout = max(int(config.training.get("updates_per_rollout", 1)), 1)
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

    if resume_dir:
        resume_path = Path(resume_dir)
        accelerator.print(f"Resuming training from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)

        sched_path = resume_path / "lr_scheduler.pt"
        if sched_path.exists():
            lr_scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))

        ckpt_name = resume_path.name
        if ckpt_name.startswith("step_"):
            try:
                global_step = int(ckpt_name.split("_")[1])
            except Exception:
                global_step = 0

        skip_batches = global_step // updates_per_rollout
        accelerator.print(f"Skipping the first {skip_batches} batches after resume.")
    else:
        accelerator.print("Starting training from scratch.")

    model.train()

    num_epochs = int(config.training.num_train_epochs)
    validation_interval = int(config.training.get("validation_interval", 0))
    checkpoint_interval = int(config.training.get("checkpoint_interval", 0))

    num_batches_per_epoch = len(train_dataloader)
    start_epoch = min(num_epochs - 1, skip_batches // max(num_batches_per_epoch, 1))
    skip_batches_in_epoch = skip_batches % max(num_batches_per_epoch, 1)
    global_batch_size = int(config.training.batch_size) * accelerator.num_processes

    sample_queue: Deque[SampleTrajectoryBundle] = deque()
    pending_stats = SamplingStats()

    for epoch in range(start_epoch, num_epochs):
        batch_iter = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for batch_idx, batch in enumerate(batch_iter):
            if epoch == start_epoch and skip_batches_in_epoch > 0 and batch_idx < skip_batches_in_epoch:
                continue
            step_records, completed, stats = collect_rollouts(
                batch,
                model,
                tokenizer,
                config,
                accelerator,
                progress_desc=f"Rollout E{epoch+1}",
            )

            gathered_step_records = accelerator.gather_object(step_records)
            gathered_completed = accelerator.gather_object(completed)
            gathered_stats = accelerator.gather_object(stats)
            gathered_counts = accelerator.gather_object(len(batch))

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

            new_samples = _filter_sample_trajectories(all_step_records, all_completed, tokenizer)
            _reindex_sample_trajectories(new_samples, 0)
            sample_queue.extend(new_samples)

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

                stats_for_logging = pending_stats
                pending_stats = SamplingStats()

                for _ in range(updates_per_rollout):
                    loss_desc = (
                        f"Loss E{epoch + 1}/{num_epochs} B{batch_idx + 1}/{len(train_dataloader)}"
                    )
                    with accelerator.accumulate(model):
                        rl_loss, clip_fraction, rl_steps = compute_losses(
                            training_step_records,
                            model,
                            tokenizer,
                            config,
                            accelerator,
                            progress_desc=loss_desc,
                        )

                        if accelerator.sync_gradients and config.training.max_grad_norm is not None:
                            accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                            optimizer.step()
                            lr_scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        global_step += 1
                        metrics = {
                            "loss/rl": rl_loss,
                            "train/clip_fraction": clip_fraction,
                            "train/raw_reward_sum": raw_reward_sum,
                            "train/rl_steps": rl_steps,
                        }

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
                            ckpt_dir = Path(project_dir) / "ckpt" / config.model.optimized_name / f"step_{global_step:06d}"
                            ckpt_dir.mkdir(parents=True, exist_ok=True)
                            save_checkpoint(
                                model,
                                tokenizer,
                                accelerator,
                                config,
                                project_dir,
                                global_step,
                                save_training_state=config.experiment.get("save_training_state", True),
                            )
                            if accelerator.is_main_process:
                                torch.save(lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt")

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
    )
    if accelerator.is_main_process:
        torch.save(lr_scheduler.state_dict(), final_ckpt / "lr_scheduler.pt")

    accelerator.end_training()


if __name__ == "__main__":
    main()
