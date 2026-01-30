#!/usr/bin/env bash
set -euo pipefail

MODEL="./save/omnicontrol_ckpt/model_humanml3d.pt"
REPS=1

OUTROOT="/home/yxc/RoboVerse/OmniControl-main/save/omnicontrol_ckpt/physics_noisy"
mkdir -p "$OUTROOT"
#define your own prompts for motion generation
prompts=(
  # # A. Contact-Unsafe Locomotion
  # "Walk forward with frequent foot slipping."
  # "Walk forward while barely lifting the feet off the ground."
  # "Walk forward with inconsistent foot contact timing."
  # "Walk forward with exaggerated heel-first landings."
  # "Walk forward with delayed foot placement."
  # "Walk forward while feet occasionally lose ground contact."

  # # B. Balance-Critical Motions
  # "Walk forward while leaning excessively forward."
  # "Walk forward while leaning excessively backward."
  # "Walk forward with unstable side-to-side balance."
  # "Walk forward with exaggerated torso sway."
  # "Walk forward while nearly losing balance each step."
  # "Walk forward with marginally stable posture."

  # # C. Over-Dynamic / Inertial Motions
  # "Rapidly accelerate from standing into walking."
  # "Walk forward and abruptly change speed."
  # "Walk forward with sudden bursts of acceleration."
  # "Walk forward and decelerate too late."
  # "Walk forward with excessive momentum."
  # "Walk forward while frequently overcorrecting motion."

  # # D. Perturbation-Recovery
  # "Walk forward and recover from an unrealistic slip."
  # "Walk forward and recover from an exaggerated stumble."
  # "Walk forward and regain balance after an abrupt push."
  # "Walk forward and recover from poorly timed disturbance."
  # "Walk forward and stabilize after an unrealistic perturbation."
  # "Walk forward with delayed balance recovery."

  # # E. Transition with Physical Ambiguity
  # "Abruptly transition from standing to fast walking."
  # "Walk forward and suddenly stop without preparation."
  # "Walk forward and immediately reverse direction."
  # "Walk forward and rotate body too quickly."
  # "Walk forward and change direction without slowing down."
  # "Walk forward with unnatural start-stop transitions."

  # # F. Upper-Body / Whole-Body Inconsistency
  # "Walk forward while swinging arms excessively."
  # "Walk forward with arms moving out of sync with legs."
  # "Walk forward with rigid upper body and active legs."
  # "Walk forward while upper body motion destabilizes balance."
  # "Walk forward with conflicting upper and lower body motion."
  # "Walk forward with unnatural whole-body coordination."

# =========================
  # G. Near-Slip but Recoverable Locomotion
  # =========================
  "Walk forward while almost slipping on each step."
  "Walk forward with subtle but repeated foot slippage."
  "Walk forward while feet briefly lose traction but recover."
  "Walk forward with marginal ground friction."

  # =========================
  # H. Marginal Stability Walking
  # =========================
  "Walk forward with barely stable posture."
  "Walk forward while constantly correcting balance."
  "Walk forward with minimal balance margin."
  "Walk forward with unstable but controlled gait."

  # =========================
  # I. Mild Disturbance Rejection
  # =========================
  "Walk forward and recover from a small unexpected push."
  "Walk forward and stabilize after a light disturbance."
  "Walk forward while handling a mild external perturbation."
  "Walk forward and smoothly regain balance after disturbance."

  # =========================
  # J. Energy-Inefficient but Feasible Motions
  # =========================
  "Walk forward with excessive joint effort."
  "Walk forward using unnecessarily strong movements."
  "Walk forward with inefficient, forceful gait."
  "Walk forward while wasting energy on motion corrections."

  # =========================
  # K. Poorly Timed Contact Transitions
  # =========================
  "Walk forward with poorly timed foot contacts."
  "Walk forward with delayed ground contact recovery."
  "Walk forward with inconsistent step timing."
  "Walk forward with mistimed foot placement."

  # =========================
  # L. Upper-Body Momentum Mismanagement
  # =========================
  "Walk forward while upper body momentum destabilizes gait."
  "Walk forward with excessive torso momentum."
  "Walk forward while upper body motion disrupts balance."
  "Walk forward with poorly controlled torso movement."
)

echo "Total prompts: ${#prompts[@]}"

# rename prompt to a slug for directory naming
slugify () {
  echo "$1" | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g; s/_+/_/g' \
    | cut -c1-80
}

i=0
for p in "${prompts[@]}"; do
  i=$((i+1))
  slug="$(slugify "$p")"
  outdir="${OUTROOT}/$(printf "%03d" "$i")_${slug}"

  echo "[$i/${#prompts[@]}] $p"
  echo "  -> $outdir"

  python -m sample.generate \
    --model_path "$MODEL" \
    --num_repetitions "$REPS" \
    --text_prompt "$p" \
    --output_dir "$outdir"

done

echo "Done. Outputs in: $OUTROOT"
