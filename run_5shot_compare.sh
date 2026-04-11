#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU_ID="${GPU_ID:-0}"

PRETRAIN_PATH="./pre_trained_model_raw/Cora.GraphCL.GCN.64_hidden_dim.pth"
COMMON_ARGS=(
  --pre_train_model_path "$PRETRAIN_PATH"
  --task NodeTask
  --dataset_name Cora
  --preprocess_method none
  --gnn_type GCN
  --shot_num 5
  --run_split 1
  --hid_dim 64
  --num_layer 2
  --epochs 100
  --seed 1 2 3 4 5
)

mkdir -p logs/RobustPrompt-T/GraphCL
mkdir -p logs/GPPT/GraphCL

run_exp() {
  local name="$1"
  shift
  echo "============================================================"
  echo "[$(date '+%F %T')] Start: $name"
  echo "GPU_ID=$GPU_ID"
  echo "============================================================"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" MyTask.py "$@" | tee "$name"
  echo "[$(date '+%F %T')] Done: $name"
}

# 1. RobustPrompt-T + GraphCL + clean + 5-shot + original
run_exp \
  "./logs/RobustPrompt-T/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_no_attack_original.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type RobustPrompt-T \
  --filter_mode original

# 2. RobustPrompt-T + GraphCL + Meta_Self-0.05 + 5-shot + original
run_exp \
  "./logs/RobustPrompt-T/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_Meta_Self_0.05_original.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type RobustPrompt-T \
  --attack_downstream \
  --attack_method Meta_Self-0.05 \
  --filter_mode original

# 3. RobustPrompt-T + GraphCL + Meta_Self-0.05 + 5-shot + neighbor_similarity
run_exp \
  "./logs/RobustPrompt-T/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_Meta_Self_0.05_neighbor_similarity.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type RobustPrompt-T \
  --attack_downstream \
  --attack_method Meta_Self-0.05 \
  --filter_mode neighbor_similarity \
  --filter_sim1_weight 0.5 \
  --filter_sim2_weight 0.5

# 4. RobustPrompt-T + GraphCL + Meta_Self-0.05 + 5-shot + hybrid
run_exp \
  "./logs/RobustPrompt-T/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_Meta_Self_0.05_hybrid.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type RobustPrompt-T \
  --attack_downstream \
  --attack_method Meta_Self-0.05 \
  --filter_mode hybrid \
  --filter_sim1_weight 0.5 \
  --filter_sim2_weight 0.5 \
  --filter_hybrid_alpha 0.5

# 5. GPPT + GraphCL + clean + 5-shot + no filter
run_exp \
  "./logs/GPPT/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_no_attack.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type GPPT

# 6. GPPT + GraphCL + Meta_Self-0.05 + 5-shot + no filter
run_exp \
  "./logs/GPPT/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_Meta_Self_0.05.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type GPPT \
  --attack_downstream \
  --attack_method Meta_Self-0.05

# 7. GPPT + GraphCL + Meta_Self-0.05 + 5-shot + original
# 注意：只有在你已经把 filter 真正接进 GPPTPrompt 后，这组才有意义。
run_exp \
  "./logs/GPPT/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_Meta_Self_0.05_original.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type GPPT \
  --attack_downstream \
  --attack_method Meta_Self-0.05 \
  --filter_mode original

# 8. GPPT + GraphCL + Meta_Self-0.05 + 5-shot + neighbor_similarity
# 注意：只有在你已经把 filter 真正接进 GPPTPrompt 后，这组才有意义。
run_exp \
  "./logs/GPPT/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_Meta_Self_0.05_neighbor_similarity.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type GPPT \
  --attack_downstream \
  --attack_method Meta_Self-0.05 \
  --filter_mode neighbor_similarity \
  --filter_sim1_weight 0.5 \
  --filter_sim2_weight 0.5

# 9. GPPT + GraphCL + Meta_Self-0.05 + 5-shot + hybrid
# 注意：只有在你已经把 filter 真正接进 GPPTPrompt 后，这组才有意义。
run_exp \
  "./logs/GPPT/GraphCL/Cora_shot_5_split_1_seed_1_2_3_4_5_Meta_Self_0.05_hybrid.log" \
  "${COMMON_ARGS[@]}" \
  --prompt_type GPPT \
  --attack_downstream \
  --attack_method Meta_Self-0.05 \
  --filter_mode hybrid \
  --filter_sim1_weight 0.5 \
  --filter_sim2_weight 0.5 \
  --filter_hybrid_alpha 0.5
