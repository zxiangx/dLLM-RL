import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List



import torch
from torch.utils.data import Dataset


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
def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


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