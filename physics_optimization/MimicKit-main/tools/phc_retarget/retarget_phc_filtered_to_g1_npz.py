import argparse
import os
import re
import sys

import joblib
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MIMICKIT_PKG = os.path.join(ROOT, "mimickit")
sys.path.insert(0, ROOT)
sys.path.insert(0, MIMICKIT_PKG)

from mimickit.anim.mjcf_char_model import MJCFCharModel  # noqa: E402
from mimickit.util import torch_util  # noqa: E402


SMPL_JOINTS_24 = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

JOINT_MATCHES = [
    ("pelvis", "Pelvis"),
    ("left_hip_pitch_link", "L_Hip"),
    ("left_knee_link", "L_Knee"),
    ("left_ankle_roll_link", "L_Ankle"),
    ("right_hip_pitch_link", "R_Hip"),
    ("right_knee_link", "R_Knee"),
    ("right_ankle_roll_link", "R_Ankle"),
    ("left_shoulder_roll_link", "L_Shoulder"),
    ("left_elbow_link", "L_Elbow"),
    ("left_wrist_yaw_link", "L_Hand"),
    ("right_shoulder_roll_link", "R_Shoulder"),
    ("right_elbow_link", "R_Elbow"),
    ("right_wrist_yaw_link", "R_Hand"),
    ("head_link", "Head"),
]


def sanitize_name(name):
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name


def y_up_to_z_up(points):
    transform = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
    if points.ndim == 1:
        return transform @ points
    return points @ transform.T


def quat_xyzw_to_exp(quat):
    q = torch.tensor(quat, dtype=torch.float32)
    exp = torch_util.quat_to_exp_map(q)
    return exp.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Retarget PHC filtered pkl to G1 MimicKit npz.")
    parser.add_argument("--input_pkl", required=True, help="amass_phc_filtered.pkl")
    parser.add_argument("--out_dir", required=True, help="Output directory for G1 npz")
    parser.add_argument("--manifest", required=True, help="Output dataset yaml")
    parser.add_argument("--char_file", required=True, help="MimicKit g1.xml")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--iters", type=int, default=200, help="Optimization iterations per clip")
    parser.add_argument("--lr", type=float, default=0.02, help="Optimizer learning rate")
    parser.add_argument("--dof_reg", type=float, default=0.01, help="Dof L2 regularization weight")
    parser.add_argument("--min_len", type=int, default=10, help="Minimum frames to keep")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of motions (0=all)")
    parser.add_argument("--y_up", action="store_true", help="Convert y-up to z-up")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output npz exists")
    args = parser.parse_args()

    data = joblib.load(args.input_pkl)
    if not isinstance(data, dict):
        raise ValueError("Expected dict in input pkl.")

    device = torch.device(args.device)
    char = MJCFCharModel(device=device)
    char.load(args.char_file)
    body_names = char.get_body_names()
    dof_size = char.get_dof_size()

    smpl_idx = [SMPL_JOINTS_24.index(name) for _, name in JOINT_MATCHES]
    g1_idx = [body_names.index(name) for name, _ in JOINT_MATCHES]

    os.makedirs(args.out_dir, exist_ok=True)
    motions = []

    count = 0
    for key, val in data.items():
        if args.limit and count >= args.limit:
            break

        joints = np.asarray(val.get("smpl_joints", None), dtype=np.float32)
        root_pos = np.asarray(val.get("root_trans_offset", None), dtype=np.float32)
        root_rot = np.asarray(val.get("root_rot", None), dtype=np.float32)
        fps = int(val.get("fps", 30))

        if joints is None or root_pos is None or root_rot is None:
            continue
        if joints.shape[0] < args.min_len:
            continue

        if args.y_up:
            joints = y_up_to_z_up(joints)
            root_pos = y_up_to_z_up(root_pos)

        if root_rot.ndim == 1:
            root_rot = np.broadcast_to(root_rot[None, :], (joints.shape[0], 4))

        root_rot_exp = quat_xyzw_to_exp(root_rot)

        target = torch.tensor(joints[:, smpl_idx, :], device=device)
        root_pos_t = torch.tensor(root_pos, device=device)
        root_rot_t = torch.tensor(root_rot_exp, device=device, requires_grad=True)
        dof = torch.zeros((joints.shape[0], dof_size), device=device, requires_grad=True)
        root_pos_offset = torch.zeros((1, 3), device=device, requires_grad=True)

        optimizer = torch.optim.Adam([dof, root_rot_t, root_pos_offset], lr=args.lr)

        for _ in range(args.iters):
            joint_rot = char.dof_to_rot(dof)
            root_quat = torch_util.exp_map_to_quat(root_rot_t)
            body_pos, body_rot = char.forward_kinematics(root_pos_t + root_pos_offset, root_quat, joint_rot)
            pred = body_pos[:, g1_idx, :]
            loss = torch.mean((pred - target) ** 2)
            loss = loss + args.dof_reg * torch.mean(dof ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            joint_rot = char.dof_to_rot(dof)
            root_quat = torch_util.exp_map_to_quat(root_rot_t)
            body_pos, body_rot = char.forward_kinematics(root_pos_t + root_pos_offset, root_quat, joint_rot)
            min_z = torch.min(body_pos[..., 2])
            body_pos[..., 2] -= min_z
            root_pos_t = root_pos_t + root_pos_offset
            root_pos_t[..., 2] -= min_z
            body_pos_np = body_pos.cpu().numpy().astype(np.float32)
            body_rot_np = body_rot.cpu().numpy().astype(np.float32)
            dof_np = dof.cpu().numpy().astype(np.float32)

        out_name = sanitize_name(key) + "_g1_jpos.npz"
        out_path = os.path.join(args.out_dir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            motions.append({"file": out_path, "weight": 1.0})
            count += 1
            continue

        np.savez(
            out_path,
            fps=fps,
            body_names=np.array(body_names, dtype=object),
            dof_positions=dof_np,
            body_positions=body_pos_np,
            body_rotations=body_rot_np,
        )

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
