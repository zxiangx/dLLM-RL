import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence
from accelerate.utils import gather_object
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from models.sampling import gumbel_noise
from train.sudoku_rl_utils import build_location_distribution, compute_logits_with_padding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test inference for Diffusion LLM")
    parser.add_argument("--model_path", required=True, help="Path to pretrained model")
    parser.add_argument("--test_file", required=True, help="Path to jsonl test file")
    parser.add_argument("--trials", type=int, default=1, help="Number of attempts per example")
    parser.add_argument("--token_temperature", type=float, default=1.0, help="Sampling temperature for tokens")
    parser.add_argument("--location_temperature", type=float, default=1.0, help="Sampling temperature for locations")
    parser.add_argument("--tokens_per_step", type=int, default=1, help="How many tokens to decode per step")
    parser.add_argument("--max_generation_length", type=int, default=256, help="Number of mask tokens to fill")
    parser.add_argument("--block_size", type=int, default=0, help="Decode masks in fixed-size blocks")
    parser.add_argument("--num_gpus", type=int, default=None, help="Expected number of GPU processes")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mask_token", type=str, default="<|mdm_mask|>", help="Mask token string")
    parser.add_argument("--pad_token", type=str, default="<|endoftext|>", help="Pad token")
    parser.add_argument("--output", type=str, default=None, help="Optional output jsonl path")
    parser.add_argument("--use_prompt_cache", action="store_true", help="Enable prompt KV cache")
    return parser.parse_args()


@dataclass
class TestSample:
    prompt: str
    test: str
    idx: int


def load_tests(path: str, trials: int) -> List[TestSample]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Test file not found: {path}")

    tests: List[TestSample] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get("prompt", "")
            test = record.get("answer", "")
            for _ in range(trials):
                tests.append(TestSample(prompt=prompt, test=test, idx=idx))
    return tests


def format_prompt(tokenizer, sample: TestSample) -> str:
    content = sample.prompt
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def mask_state_from_prompt(tokenizer, prompt_text: str, max_gen: int, mask_token_id: int) -> torch.Tensor:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    filled_ids = list(prompt_ids)
    filled_ids.extend([mask_token_id] * max_gen)
    return torch.tensor(filled_ids, dtype=torch.long), len(prompt_ids)


def gumbel_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    noise = gumbel_noise(logits)
    values, indices = torch.topk(logits + noise, k=k)
    finite_mask = torch.isfinite(values)
    return indices[finite_mask]


def select_block_mask(candidate_mask: torch.Tensor, block_size: int) -> torch.Tensor:
    if block_size is None or block_size <= 0:
        return candidate_mask

    indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        return candidate_mask

    first = indices[0].item()
    block_start = (first // block_size) * block_size
    block_end = block_start + block_size
    chosen = indices[(indices >= block_start) & (indices < block_end)]
    block_mask = torch.zeros_like(candidate_mask)
    block_mask[chosen] = True
    return block_mask


def decode_batch(
    model,
    tokenizer,
    samples: Sequence[TestSample],
    device,
    token_temperature: float,
    location_temperature: float,
    tokens_per_step: int,
    max_generation_length: int,
    block_size: int,
    prompt_cache=None,
    mask_token_id: int = None,
) -> List[str]:
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) if mask_token_id is None else mask_token_id
    results: List[str] = []

    states: List[torch.Tensor] = []
    prompt_lengths: List[int] = []
    for sample in samples:
        prompt_text = format_prompt(tokenizer, sample)
        state, prompt_len = mask_state_from_prompt(tokenizer, prompt_text, max_generation_length, mask_token_id)
        states.append(state.to(device))
        prompt_lengths.append(prompt_len)

    unfinished = [True] * len(states)
    with torch.no_grad():
        cnt = 0
        while any(unfinished):
            cnt += 1
            logits_batch = compute_logits_with_padding(
                states,
                model,
                tokenizer,
                device,
                prompt_lengths=prompt_lengths,
                prompt_cache=prompt_cache,
                allow_cache_in_grad=False,
            )
            for idx, logits in enumerate(logits_batch):
                if not unfinished[idx]:
                    continue
                logits = logits.to(torch.float32)
                state = states[idx]
                candidate_mask = state.eq(mask_token_id)
                if not torch.any(candidate_mask):
                    unfinished[idx] = False
                    continue

                block_mask = select_block_mask(candidate_mask, block_size)
                _, loc_scores = build_location_distribution(
                    logits,
                    candidate_mask=block_mask,
                    epsilon=1e-9,
                    temperature=location_temperature,
                )

                decode_count = min(tokens_per_step, int(block_mask.sum().item()))
                if decode_count <= 0:
                    unfinished[idx] = False
                    continue

                loc_logits = torch.where(block_mask, loc_scores, torch.full_like(loc_scores, -1e9))
                sampled_locations = gumbel_topk(loc_logits, decode_count)

                for loc in sampled_locations.tolist():
                    token_logits = logits[loc] / max(token_temperature, 1e-8)
                    token_id = int(torch.argmax(token_logits + gumbel_noise(token_logits)).item())
                    state[loc] = token_id

                states[idx] = state
                if not torch.any(state.eq(mask_token_id)):
                    unfinished[idx] = False

    for state, prompt_len in zip(states, prompt_lengths):
        gen_tokens = state[prompt_len:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        results.append(text)
    return results


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(split_batches=True)
    if args.num_gpus is not None and args.num_gpus != accelerator.num_processes:
        raise ValueError(
            f"Requested {args.num_gpus} GPUs but running with {accelerator.num_processes} processes; launch with accelerate."
        )

    torch.manual_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.to(accelerator.device)
    model.eval()

    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": args.mask_token, "pad_token": args.pad_token})
        model.resize_token_embeddings(len(tokenizer))

    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    all_tests = load_tests(args.test_file, args.trials)

    prompt_cache = None
    if args.use_prompt_cache:
        from train.sudoku_rl_utils import PromptCacheManager
        prompt_cache = PromptCacheManager()

    local_outputs: List[Dict[str, Any]] = []
    # 这里用上下文管理器拿到当前进程的数据切片
    with accelerator.split_between_processes(all_tests) as shard:
        for start in tqdm(
            range(0, len(shard), args.batch_size),
            disable=not accelerator.is_main_process,
        ):
            batch_samples = shard[start : start + args.batch_size]
            outputs = decode_batch(
                model,
                tokenizer,
                batch_samples,
                accelerator.device,
                token_temperature=args.token_temperature,
                location_temperature=args.location_temperature,
                tokens_per_step=args.tokens_per_step,
                max_generation_length=args.max_generation_length,
                block_size=args.block_size,
                prompt_cache=prompt_cache,
                mask_token_id=mask_token_id,
            )

            for sample, text in zip(batch_samples, outputs):
                local_outputs.append(
                    {"idx": sample.idx, "prompt": sample.prompt, "answer": sample.test, "output": text}
                )
    gathered = gather_object(local_outputs)

    if accelerator.is_main_process:
        gathered.sort(key=lambda x: x["idx"])
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                for record in gathered:
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            for record in gathered:
                print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()