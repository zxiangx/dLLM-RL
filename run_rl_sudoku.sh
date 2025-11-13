#!/bin/bash
# ============================
#  RL Sudoku (Accelerate + DeepSpeed) Launcher
# ============================

#############################
# 配置变量（在这里修改即可）
#############################

NUM_MACHINES=1
MACHINE_RANK=0
MAIN_IP="127.0.0.1"
MAIN_PORT=29500
export TORCH_CUDA_ARCH_LIST="8.0"

DEEPSPEED_CONFIG="accelerate_configs/1_node_8_gpus_deepspeed_zero3.yaml"
TRAIN_SCRIPT="train/rl_sudoku_llada.py"
CONFIG_FILE="configs/rl_sudoku_llada.yaml"

#############################
# 打印执行信息
#############################

echo "Launching training with Accelerate..."
echo "-----------------------------------------"
echo " NUM_MACHINES     = ${NUM_MACHINES}"
echo " MACHINE_RANK     = ${MACHINE_RANK}"
echo " MAIN IP:PORT     = ${MAIN_IP}:${MAIN_PORT}"
echo " DS CONFIG        = ${DEEPSPEED_CONFIG}"
echo " TRAIN SCRIPT     = ${TRAIN_SCRIPT}"
echo " CONFIG FILE      = ${CONFIG_FILE}"
echo "-----------------------------------------"
echo

#############################
# 运行 Accelerate
#############################

accelerate launch \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_IP}" \
  --main_process_port "${MAIN_PORT}" \
  --config_file "${DEEPSPEED_CONFIG}" \
  "${TRAIN_SCRIPT}" \
  config="${CONFIG_FILE}"
