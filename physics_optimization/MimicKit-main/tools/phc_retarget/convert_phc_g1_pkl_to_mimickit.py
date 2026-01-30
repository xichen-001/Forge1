import argparse
import os
import re
import sys

import joblib
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from mimickit.anim import motion as motion_lib  # noqa: E402
from mimickit.util import torch_util  # noqa: E402


PHC_G1_DOF_NAMES = [
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",  # waist_pitch_joint in MimicKit
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


def sanitize_name(name):
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name


def quat_to_exp_map_np(quat):
    q = torch.tensor(quat, dtype=torch.float32)
    exp = torch_util.quat_to_exp_map(q)
    return exp.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Convert PHC G1 retargeted pkl to MimicKit motions.")
    parser.add_argument("--input_pkl", required=True, help="PHC amass_all.pkl path")
    parser.add_argument("--out_dir", required=True, help="Output directory for MimicKit pkl motions")
    parser.add_argument("--manifest", required=True, help="Output dataset yaml")
    parser.add_argument("--min_len", type=int, default=10, help="Minimum frames to keep")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of motions (0=all)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output pkl exists")
    args = parser.parse_args()

    data = joblib.load(args.input_pkl)
    if not isinstance(data, dict):
        raise ValueError("Expected a dict in input pkl.")

    os.makedirs(args.out_dir, exist_ok=True)
    motions = []

    count = 0
    for key, val in data.items():
        if args.limit and count >= args.limit:
            break

        root_pos = np.asarray(val.get("root_trans_offset", None), dtype=np.float32)
        root_rot = np.asarray(val.get("root_rot", None), dtype=np.float32)
        dof = np.asarray(val.get("dof", None), dtype=np.float32)
        fps = int(val.get("fps", 30))

        if root_pos is None or root_rot is None or dof is None:
            continue
        if root_pos.shape[0] < args.min_len:
            continue
        if dof.shape[1] != len(PHC_G1_DOF_NAMES):
            continue

        exp_map = quat_to_exp_map_np(root_rot)
        frames = np.concatenate([root_pos, exp_map, dof], axis=-1).astype(np.float32)

        out_name = sanitize_name(key) + ".pkl"
        out_path = os.path.join(args.out_dir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            motions.append({"file": out_path, "weight": 1.0})
            count += 1
            continue

        motion = motion_lib.Motion(loop_mode=motion_lib.LoopMode.CLAMP, fps=fps, frames=frames)
        motion.save(out_path)

        motions.append({"file": out_path, "weight": 1.0})
        count += 1

    with open(args.manifest, "w", encoding="utf-8") as f:
        f.write("motions:\n")
        for item in motions:
            f.write(f"- file: {item['file']}\n")
            f.write(f"  weight: {item['weight']}\n")

    print(f"Saved {len(motions)} motions to {args.manifest}")


if __name__ == "__main__":
    main()
