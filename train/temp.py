def compute_losses(
    step_records: List[StepRecord],
    model: LLaDAModelLM,
    tokenizer,
    config,
    device: torch.device,
    accelerator: Accelerator,
    progress_desc: str = "",
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

    # --- 进度条：按前向次数来算 ---
    num_forward_batches = math.ceil(len(step_records) / rollout_batch_size)
    loss_pbar = None
    if accelerator.is_local_main_process and num_forward_batches > 0:
        desc = progress_desc or "ComputeLoss"
        loss_pbar = tqdm(
            total=num_forward_batches,
            desc=desc,
            leave=False,
            disable=not accelerator.is_local_main_process,
        )

    for start in range(0, len(step_records), rollout_batch_size):
        batch_records = step_records[start : start + rollout_batch_size]
        states_before = torch.stack([step.state_before for step in batch_records]).to(device)

        logits_before = model(states_before).logits

        # 更新进度条：一次前向就是一次单位
        if loss_pbar is not None:
            loss_pbar.update(1)

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

            clip_ratio = torch.clamp(
                ratio,
                1 - training_cfg.clip_epsilon,
                1 + training_cfg.clip_epsilon,
            )

            if (clip_ratio - ratio).abs() > 1e-8:
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
            # 这里的 sft_loss 写法不变
            sft_losses.append(-ratio.detach() * f_theta_val)

    if loss_pbar is not None:
        loss_pbar.close()

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
            # shift opt, to guarantee reward = state_after.f_value
            rewards = rewards[1:] + [0.0]
            reward_rows.append(pad_to_length(rewards, max_steps))

        rewards_tensor = torch.stack(reward_rows, dim=0).to(device)

        returns = torch.zeros_like(rewards_tensor)
        running = torch.zeros(rewards_tensor.size(0), device=device)

        for t in reversed(range(max_steps)):
            running = rewards_tensor[:, t] + gamma * running
            returns[:, t] = running

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
