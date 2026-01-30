#  Command guided

#  Train a teacher policy using command guidance
export WANDB_PROJECT=g1_teacher
export WANDB_ENTITY=your_entity   # 没有就不设
export WANDB_RUN_NAME=stage1


# motion data filtering for currculum learning
cd /home/yuan/Forge-main/physics_optimization/MimicKit-main

# stage1
python tools/g1_amass/build_manifest.py \
  --root data/motions/g1_phc_filtered \
  --out data/datasets/g1_phc_stage1.yaml \
  --mjcf data/assets/g1/g1.xml \
  --stage locomotion \
  --min_len 0.4 --max_len 14.0 \
  --min_root_z 0.20 --max_root_z 1.8 \
  --max_root_roll 70 --max_root_pitch 70 --max_root_tilt_ratio 0.6 \
  --max_root_speed 8.0 --max_root_ang_speed 18.0 \
  --max_dof_vel 20.0 --max_dof_acc 160.0 \
  --max_dof_vel_ratio 0.08 --max_dof_acc_ratio 0.08 \
  --contact_height 0.12 --contact_vz 0.4 \
  --max_foot_slide 0.35 --min_contact_ratio 0.01 \
  --max_contact_switch_rate 15.0 \
  --max_com_dist 0.80 --max_com_ratio 0.8 \
  --stab_eps 0.80 --max_unstable_len 500 \
  --require_z_up --min_up_dot 0.0 --max_bad_up_ratio 0.3 \
  --weight_by_score

# stage2
python tools/g1_amass/build_manifest.py \
  --root data/motions/g1_phc_filtered \
  --out data/datasets/g1_phc_stage2.yaml \
  --mjcf data/assets/g1/g1.xml \
  --stage all \
  --min_len 0.4 --max_len 16.0 \
  --min_root_z 0.20 --max_root_z 2.0 \
  --max_root_roll 75 --max_root_pitch 75 --max_root_tilt_ratio 0.7 \
  --max_root_speed 10.0 --max_root_ang_speed 20.0 \
  --max_dof_vel 22.0 --max_dof_acc 180.0 \
  --max_dof_vel_ratio 0.12 --max_dof_acc_ratio 0.12 \
  --contact_height 0.14 --contact_vz 0.45 \
  --max_foot_slide 0.45 --min_contact_ratio 0.02 \
  --max_contact_switch_rate 20.0 \
  --max_com_dist 1.2 --max_com_ratio 0.8 \
  --stab_eps 1.0 --max_unstable_len 600 \
  --require_z_up --min_up_dot 0.0 --max_bad_up_ratio 0.4 \
  --weight_by_score

# stage3
python tools/g1_amass/build_manifest.py \
  --root data/motions/g1_phc_filtered \
  --out data/datasets/g1_phc_stage3.yaml \
  --mjcf data/assets/g1/g1.xml \
  --stage all \
  --allow_ground_motion \
  --min_len 0.3 --max_len 20.0 \
  --min_root_z 0.05 --max_root_z 2.5 \
  --max_root_roll 90 --max_root_pitch 90 --max_root_tilt_ratio 1.0 \
  --max_root_speed 14.0 --max_root_ang_speed 28.0 \
  --max_dof_vel 35.0 --max_dof_acc 260.0 \
  --max_dof_vel_ratio 0.25 --max_dof_acc_ratio 0.25 \
  --contact_height 0.20 --contact_vz 0.7 \
  --max_foot_slide 1.0 --min_contact_ratio 0.0 \
  --max_contact_switch_rate 50.0 \
  --max_com_dist 3.0 --max_com_ratio 1.0 \
  --stab_eps 1.2 --max_unstable_len 900 \
  --require_z_up --min_up_dot -0.2 --max_bad_up_ratio 0.6 \
  --weight_by_score


# Train teacher policy for stage 1-3 sequentially

#change motion file in env config：data/envs/amp_g1_env.yaml
python - <<'PY'
import yaml
env_path = "data/envs/amp_g1_env.yaml"
motion_file = "data/datasets/g1_phc_stage1.yaml"
with open(env_path, "r") as f:
    cfg = yaml.safe_load(f)
cfg["motion_file"] = motion_file
with open(env_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("motion_file ->", motion_file)
PY


# Training 
python mimickit/run.py \
  --arg_file args/amp_g1_args.txt \
  --num_envs 8192 \
  --out_dir output/teacher_g1_stage1 \
  --visualize false \
  --logger wandb

#stage2,3 is similar, just change the motion_file and out_dir accordingly.

# pull config from ssh to local for record


#find the latest 10 model.pt files
cd /home/yuan/Forge-main/physics_optimization/MimicKit-main
find output -name model.pt -printf '%T@ %p\n' | sort -nr | head -n 10

#model.pt
scp yuan@192.168.123.207:/home/yuan/Forge-main/physics_optimization/MimicKit-main/output/teacher_g1_stage1/model.pt \
  /home/yxc/RoboVerse/OmniControl-main/MimicKit-main/output/teacher_g1_stage1/model.pt

#model config files
scp yuan@192.168.123.207:/home/yuan/Forge-main/physics_optimization/MimicKit-main/output/teacher_g1_stage1/env_config.yaml \
  /home/yxc/RoboVerse/OmniControl-main/MimicKit-main/output/teacher_g1_stage1/env_config.yaml

scp yuan@192.168.123.207:/home/yuan/Forge-main/physics_optimization/MimicKit-main/output/teacher_g1_stage1/agent_config.yaml \
  /home/yxc/RoboVerse/OmniControl-main/MimicKit-main/output/teacher_g1_stage1/agent_config.yaml

scp yuan@192.168.123.207:/home/yuan/Forge-main/physics_optimization/MimicKit-main/output/teacher_g1_stage1/engine_config.yaml \
  /home/yxc/RoboVerse/OmniControl-main/MimicKit-main/output/teacher_g1_stage1/engine_config.yaml
