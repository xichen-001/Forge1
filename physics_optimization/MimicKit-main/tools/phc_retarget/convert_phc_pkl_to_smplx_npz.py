import argparse
import os
import re

import joblib
import numpy as np

try:
    from holosoma_retargeting.config_types.data_type import SMPLX_DEMO_JOINTS
except Exception as exc:
    raise RuntimeError(
        "Failed to import holosoma_retargeting. Run this script with "
        "PYTHONPATH=/home/yuan/holosoma-main/src or install the package."
    ) from exc

try:
    from holosoma_retargeting.src.utils import transform_y_up_to_z_up
except Exception:
    def transform_y_up_to_z_up(points):
        transform_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        if points.ndim == 1:
            return transform_matrix @ points
        return points @ transform_matrix.T


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


def sanitize_name(name):
    name = name.strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name


def main():
    parser = argparse.ArgumentParser(description="Convert PHC filtered PKL to SMPL-X style NPZ.")
    parser.add_argument("--input_pkl", required=True, help="Path to amass_phc_filtered.pkl")
    parser.add_argument("--out_dir", required=True, help="Output directory for smplx npz files")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of motions (0 = all)")
    parser.add_argument("--y_up", action="store_true", help="Convert y-up to z-up")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output npz exists")
    args = parser.parse_args()

    data = joblib.load(args.input_pkl)
    if not isinstance(data, dict):
        raise ValueError("Expected dict in pkl.")

    os.makedirs(args.out_dir, exist_ok=True)

    idx_map = [SMPL_JOINTS_24.index(name) for name in SMPLX_DEMO_JOINTS]

    count = 0
    for key, val in data.items():
        if args.limit and count >= args.limit:
            break
        joints = val.get("smpl_joints", None)
        if joints is None:
            continue
        joints = np.asarray(joints, dtype=np.float32)
        if joints.shape[1] != len(SMPL_JOINTS_24):
            continue
        joints = joints[:, idx_map, :]

        if args.y_up:
            joints = transform_y_up_to_z_up(joints)

        z = joints[..., 2]
        height = float(np.percentile(z, 95) - np.percentile(z, 5))
        if height <= 0:
            continue

        fps = int(val.get("fps", 30))

        out_name = sanitize_name(key) + ".npz"
        out_path = os.path.join(args.out_dir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            count += 1
            continue

        np.savez(
            out_path,
            global_joint_positions=joints,
            height=height,
            fps=fps,
        )
        count += 1

    print(f"Saved {count} motions to {args.out_dir}")


if __name__ == "__main__":
    main()
