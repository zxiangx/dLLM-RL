import os
import subprocess
from typing import Any, List

from omegaconf import OmegaConf


def _collect_cli_overrides(cli_conf) -> List[str]:
    overrides: List[str] = []

    def _recurse(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else key
                _recurse(next_prefix, inner)
        elif isinstance(value, list):
            for idx, inner in enumerate(value):
                next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                _recurse(next_prefix, inner)
        else:
            overrides.append(f"{prefix}={value}")

    container = OmegaConf.to_container(cli_conf, resolve=True)
    for key, value in container.items():
        if key == "config":
            continue
        _recurse(key, value)

    return overrides


def main() -> None:
    cli_conf = OmegaConf.from_cli()
    if "config" not in cli_conf:
        raise ValueError("Please provide the config path as `config=...` on the command line")

    config_path = cli_conf.config
    yaml_conf = OmegaConf.load(config_path)
    config = OmegaConf.merge(yaml_conf, cli_conf)

    ds_file = config.experiment.get("deepspeed_file", "1_node_8_gpus_deepspeed_zero3")
    num_machines = int(config.experiment.get("num_machines", 1))
    machine_rank = int(config.experiment.get("machine_rank", 0))
    main_ip = str(config.experiment.get("main_process_ip", "127.0.0.1"))
    main_port = str(config.experiment.get("main_process_port", 29500))

    accelerate_cmd = [
        "accelerate",
        "launch",
        "--num_machines",
        str(num_machines),
        "--machine_rank",
        str(machine_rank),
        "--main_process_ip",
        main_ip,
        "--main_process_port",
        main_port,
        "--config_file",
        f"accelerate_configs/{ds_file}.yaml",
        "train/rl_sudoku_llada.py",
        f"config={config_path}",
    ]

    overrides = _collect_cli_overrides(cli_conf)
    accelerate_cmd.extend(overrides)

    env = os.environ.copy()
    env.setdefault("DS_SKIP_CUDA_CHECK", "1")

    subprocess.run(" ".join(accelerate_cmd), shell=True, check=True, env=env)


if __name__ == "__main__":
    main()