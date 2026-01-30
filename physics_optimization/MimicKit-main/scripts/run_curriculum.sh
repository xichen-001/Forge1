#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yuan/Forge-main/physics_optimization/MimicKit-main"
ARGS="$ROOT/args/amp_g1_args.txt"
ENV_CFG="$ROOT/data/envs/amp_g1_env.yaml"
NUM_ENVS=20000
STEPS_PER_ITER=16

STAGE1="$ROOT/data/datasets/g1_phc_stage1.yaml"
STAGE2="$ROOT/data/datasets/g1_phc_stage2.yaml"
STAGE3="$ROOT/data/datasets/g1_phc_stage3.yaml"

OUT1="$ROOT/output/teacher_g1_stage1"
OUT2="$ROOT/output/teacher_g1_stage2"
OUT3="$ROOT/output/teacher_g1_stage3"

START_CKPT="$ROOT/output/teacher_g1_stage1/model.pt"

run_stage () {
  local stage_name="$1"
  local motion_file="$2"
  local out_dir="$3"
  local ckpt="$4"
  local iters="$5"
  local max_samples=$((NUM_ENVS * STEPS_PER_ITER * iters))

  python - <<PY
import yaml
p="$ENV_CFG"
c=yaml.safe_load(open(p))
c["motion_file"]="$motion_file"
yaml.safe_dump(c, open(p,"w"), sort_keys=False)
print("motion_file ->", c["motion_file"])
PY

  echo "=== Running $stage_name ==="
  if [[ -f "$ckpt" ]]; then
    python "$ROOT/mimickit/run.py" \
      --arg_file "$ARGS" \
      --num_envs "$NUM_ENVS" \
      --visualize false \
      --out_dir "$out_dir" \
      --save_int_models true \
      --logger wandb \
      --model_file "$ckpt" \
      --max_samples "$max_samples"
  else
    python "$ROOT/mimickit/run.py" \
      --arg_file "$ARGS" \
      --num_envs "$NUM_ENVS" \
      --visualize false \
      --out_dir "$out_dir" \
      --save_int_models true \
      --logger wandb \
      --max_samples "$max_samples"
  fi
}

run_stage "stage1" "$STAGE1" "$OUT1" "$START_CKPT" 1000
run_stage "stage2" "$STAGE2" "$OUT2" "$OUT1/model.pt" 2000
run_stage "stage3" "$STAGE3" "$OUT3" "$OUT2/model.pt" 3500
