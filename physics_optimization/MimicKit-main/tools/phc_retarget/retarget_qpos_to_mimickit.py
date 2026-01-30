import argparse
import glob
import os

import numpy as np
import torch
import yaml

import sys

sys.path.append(".")

from mimickit.anim.motion import Motion, LoopMode
from mimickit.util.torch_util import quat_to_exp_map


def main():
    parser = argparse.ArgumentParser(description="Convert holosoma retargeted qpos npz to MimicKit pkl.")
    parser.add_argument("--input_dir", required=True, help="Directory of holosoma qpos npz files")
    parser.add_argument("--out_dir", required=True, help="Output directory for MimicKit pkl files")
    parser.add_argument("--manifest", default="", help="Optional output manifest yaml path")
    parser.add_argument("--dof", type=int, default=29, help="Robot DOF count")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output motions")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of motions (0 = all)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if output pkl exists")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, "*.npz")))
    if args.limit:
        files = files[: args.limit]

    motions = []
    for npz_path in files:
        data = np.load(npz_path)
        if "qpos" not in data:
            continue
        qpos = data["qpos"]
        if qpos.shape[1] < 7 + args.dof:
            continue

        root_pos = qpos[:, 0:3]
        root_rot_wxyz = qpos[:, 3:7]
        root_rot_xyzw = np.concatenate([root_rot_wxyz[:, 1:4], root_rot_wxyz[:, 0:1]], axis=-1)
        dof_pos = qpos[:, 7 : 7 + args.dof]

        root_rot_exp = quat_to_exp_map(torch.tensor(root_rot_xyzw, dtype=torch.float32)).numpy()
        frames = np.concatenate([root_pos, root_rot_exp, dof_pos], axis=-1).astype(np.float32)

        out_name = os.path.splitext(os.path.basename(npz_path))[0] + ".pkl"
        out_path = os.path.join(args.out_dir, out_name)
        if args.skip_existing and os.path.exists(out_path):
            motions.append({"file": out_path, "weight": 1.0})
            continue

        motion = Motion(loop_mode=LoopMode.CLAMP, fps=args.fps, frames=frames)
        motion.save(out_path)
        motions.append({"file": out_path, "weight": 1.0})

    if args.manifest:
        with open(args.manifest, "w") as f:
            yaml.safe_dump({"motions": motions}, f)
        print(f"Wrote manifest: {args.manifest} ({len(motions)} motions)")


if __name__ == "__main__":
    main()
