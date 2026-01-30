import argparse
import fnmatch
import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def iter_npz_files(root, pattern):
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if fnmatch.fnmatch(name, pattern):
                yield os.path.join(dirpath, name)


def load_meta(path):
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    fps = data["fps"]
    if np.ndim(fps) > 0:
        fps = float(fps.reshape(-1)[0])
    else:
        fps = float(fps)
    body_names = [str(x) for x in data["body_names"]]
    dof_names = None
    if "dof_names" in data:
        dof_names = [str(x) for x in data["dof_names"]]
    body_pos = np.asarray(data["body_positions"], dtype=np.float32)
    body_rot = np.asarray(data["body_rotations"], dtype=np.float32)
    dof_pos = np.asarray(data["dof_positions"], dtype=np.float32)
    dof_vel = None
    if "dof_velocities" in data:
        dof_vel = np.asarray(data["dof_velocities"], dtype=np.float32)
    body_vel = None
    if "body_linear_velocities" in data:
        body_vel = np.asarray(data["body_linear_velocities"], dtype=np.float32)
    body_ang_vel = None
    if "body_angular_velocities" in data:
        body_ang_vel = np.asarray(data["body_angular_velocities"], dtype=np.float32)
    return {
        "fps": fps,
        "body_names": body_names,
        "dof_names": dof_names,
        "body_pos": body_pos,
        "body_rot": body_rot,
        "dof_pos": dof_pos,
        "dof_vel": dof_vel,
        "body_vel": body_vel,
        "body_ang_vel": body_ang_vel,
    }


def load_joint_limits(mjcf_path, urdf_path):
    limits = {}
    if mjcf_path:
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        for joint in root.findall(".//joint"):
            name = joint.get("name")
            rng = joint.get("range")
            if name and rng:
                parts = [float(x) for x in rng.split()]
                if len(parts) == 2:
                    limits[name] = (parts[0], parts[1])
    if urdf_path:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for joint in root.findall(".//joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if name and limit is not None:
                lower = limit.get("lower")
                upper = limit.get("upper")
                if lower is not None and upper is not None:
                    limits[name] = (float(lower), float(upper))
    return limits


def _find_body_indices(body_names, candidates):
    indices = []
    for name in candidates:
        if name in body_names:
            indices.append(body_names.index(name))
    return indices


def _finite_diff(x, fps):
    if x.shape[0] <= 1:
        return np.zeros_like(x)
    dx = np.diff(x, axis=0) * fps
    last = dx[-1:]
    return np.concatenate([dx, last], axis=0)


def _quat_to_euler_xyzw(q):
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw


def _quat_rotate(q, v):
    qvec = q[..., :3]
    qw = q[..., 3:4]
    v = np.asarray(v, dtype=q.dtype)
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (qw * uv + uuv)


def _compute_contact(foot_pos, foot_vel, contact_height, contact_vz):
    z = foot_pos[..., 2]
    vz = foot_vel[..., 2]
    return (z < contact_height) & (np.abs(vz) < contact_vz)


def _support_dist(com_xy, foot_xy, contact_mask):
    if contact_mask.ndim == 1:
        contact_mask = contact_mask[:, None]
    contact_any = np.any(contact_mask, axis=-1)
    if not np.any(contact_any):
        return None
    support_xy = np.zeros_like(com_xy)
    for i in range(contact_mask.shape[0]):
        mask = contact_mask[i]
        if np.any(mask):
            support_xy[i] = np.mean(foot_xy[i, mask], axis=0)
        else:
            support_xy[i] = com_xy[i]
    dist = np.linalg.norm(com_xy - support_xy, axis=-1)
    return dist


def _support_dist_with_contact(com_xy, foot_xy, contact_mask, invalid_val=np.inf):
    if contact_mask.ndim == 1:
        contact_mask = contact_mask[:, None]
    contact_any = np.any(contact_mask, axis=-1)
    support_xy = np.zeros_like(com_xy)
    for i in range(contact_mask.shape[0]):
        mask = contact_mask[i]
        if np.any(mask):
            support_xy[i] = np.mean(foot_xy[i, mask], axis=0)
        else:
            support_xy[i] = com_xy[i]
    dist = np.linalg.norm(com_xy - support_xy, axis=-1)
    dist = dist.astype(np.float32)
    dist[~contact_any] = invalid_val
    return dist, contact_any


def _longest_run(mask):
    longest = 0
    curr = 0
    for val in mask:
        if val:
            curr += 1
        else:
            longest = max(longest, curr)
            curr = 0
    return max(longest, curr)


def passes_filters(meta, limits, cfg):
    body_pos = meta["body_pos"]
    body_rot = meta["body_rot"]
    dof_pos = meta["dof_pos"]
    dof_vel = meta["dof_vel"]
    body_vel = meta["body_vel"]
    body_ang_vel = meta["body_ang_vel"]
    fps = meta["fps"]
    body_names = meta["body_names"]
    dof_names = meta["dof_names"]

    if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
        return False, None
    if body_rot.ndim != 3 or body_rot.shape[-1] != 4:
        return False, None
    if dof_pos.ndim != 2:
        return False, None
    if not np.isfinite(body_rot).all() or not np.isfinite(dof_pos).all():
        return False, None

    num_frames = body_pos.shape[0]
    length = (num_frames - 1) / fps if num_frames > 1 else 0.0
    if length < cfg["min_len"] or length > cfg["max_len"]:
        return False, None

    pelvis_idx = body_names.index("pelvis") if "pelvis" in body_names else 0
    root_pos = body_pos[:, pelvis_idx, :]
    root_q = body_rot[:, pelvis_idx, :]

    if not np.isfinite(root_pos).all() or not np.isfinite(root_q).all():
        return False, None

    root_z = root_pos[:, 2]
    if root_z.min() < cfg["min_root_z"] or root_z.max() > cfg["max_root_z"]:
        return False, None

    q_norm = np.linalg.norm(root_q, axis=-1)
    if q_norm.min() < 0.5 or q_norm.max() > 1.5:
        return False, None

    if cfg["require_z_up"]:
        up = _quat_rotate(root_q, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        up_dot = up[..., 2]
        bad = up_dot < cfg["min_up_dot"]
        if np.mean(bad) > cfg["max_bad_up_ratio"]:
            return False, None

    root_vel = None
    if body_vel is not None:
        root_vel = body_vel[:, pelvis_idx, :]
    else:
        root_vel = _finite_diff(root_pos, fps)
    root_speed = np.linalg.norm(root_vel, axis=-1)
    if root_speed.max() > cfg["max_root_speed"]:
        return False, None

    if cfg["stage_min_speed"] is not None or cfg["stage_max_speed"] is not None:
        min_s = cfg["stage_min_speed"] if cfg["stage_min_speed"] is not None else 0.0
        max_s = cfg["stage_max_speed"] if cfg["stage_max_speed"] is not None else 1e9
        if np.percentile(root_speed, 95) < min_s or np.percentile(root_speed, 95) > max_s:
            return False, None

    if body_ang_vel is not None and cfg["max_root_ang_speed"] is not None:
        root_ang = body_ang_vel[:, pelvis_idx, :]
        if np.linalg.norm(root_ang, axis=-1).max() > cfg["max_root_ang_speed"]:
            return False, None

    if dof_vel is None:
        dof_vel = _finite_diff(dof_pos, fps)

    dof_acc = _finite_diff(dof_vel, fps)
    if cfg["max_dof_vel"] is not None:
        exceed = np.abs(dof_vel) > cfg["max_dof_vel"]
        if np.mean(exceed) > cfg["max_dof_vel_ratio"]:
            return False, None
    if cfg["max_dof_acc"] is not None:
        exceed = np.abs(dof_acc) > cfg["max_dof_acc"]
        if np.mean(exceed) > cfg["max_dof_acc_ratio"]:
            return False, None

    if limits and dof_names:
        margin = np.deg2rad(cfg["joint_limit_margin_deg"])
        max_ratio = cfg["joint_limit_ratio"]
        for j, name in enumerate(dof_names):
            if name in limits:
                lower, upper = limits[name]
                low = lower + margin
                high = upper - margin
                if low >= high:
                    low, high = lower, upper
                exceed = (dof_pos[:, j] < low) | (dof_pos[:, j] > high)
                if np.mean(exceed) > max_ratio:
                    return False, None

    if not cfg["allow_ground_motion"]:
        nominal = np.percentile(root_z, 90)
        if nominal <= 0:
            return False, None
        low = root_z < (cfg["min_root_ratio"] * nominal)
        if np.mean(low) > cfg["max_low_root_ratio"]:
            return False, None
        roll, pitch, _ = _quat_to_euler_xyzw(root_q)
        roll = np.degrees(roll)
        pitch = np.degrees(pitch)
        tilt = (np.abs(roll) > cfg["max_root_roll"]) | (np.abs(pitch) > cfg["max_root_pitch"])
        if np.mean(tilt) > cfg["max_root_tilt_ratio"]:
            return False, None

    foot_ids = _find_body_indices(body_names, cfg["foot_names"])
    if len(foot_ids) == 0:
        return False, None
    foot_pos = body_pos[:, foot_ids, :]
    if body_vel is not None:
        foot_vel = body_vel[:, foot_ids, :]
    else:
        foot_vel = _finite_diff(foot_pos, fps)

    contact = _compute_contact(foot_pos, foot_vel, cfg["contact_height"], cfg["contact_vz"])
    contact_ratio = np.mean(contact)
    if contact_ratio < cfg["min_contact_ratio"]:
        return False, None

    foot_vel_xy = np.linalg.norm(foot_vel[..., :2], axis=-1)
    slide_mask = contact
    if np.any(slide_mask):
        slide_vals = foot_vel_xy[slide_mask]
        slide_p95 = np.percentile(slide_vals, 95)
        if slide_p95 > cfg["max_foot_slide"]:
            return False, None

    if cfg["max_contact_switch_rate"] is not None:
        switches = np.abs(np.diff(contact.astype(np.int32), axis=0)).sum(axis=-1)
        rate = np.mean(switches) * fps
        if rate > cfg["max_contact_switch_rate"]:
            return False, None

    if cfg["max_com_dist"] is not None:
        com_xy = root_pos[:, :2]
        foot_xy = foot_pos[..., :2]
        dist = _support_dist(com_xy, foot_xy, contact)
        if dist is not None:
            if np.mean(dist > cfg["max_com_dist"]) > cfg["max_com_ratio"]:
                return False, None

    if cfg["stab_eps"] is not None:
        com_xy = root_pos[:, :2]
        foot_xy = foot_pos[..., :2]
        stab_dist, contact_any = _support_dist_with_contact(com_xy, foot_xy, contact)
        stable = stab_dist < cfg["stab_eps"]
        if stable.shape[0] == 0:
            return False, None
        if not stable[0] or not stable[-1]:
            return False, None
        unstable = ~stable
        longest_unstable = _longest_run(unstable.tolist())
        if longest_unstable >= cfg["max_unstable_len"]:
            return False, None

    score = 1.0
    if cfg["weight_by_score"]:
        terms = []
        if cfg["max_foot_slide"] is not None:
            slide_score = 1.0 - min(1.0, slide_p95 / cfg["max_foot_slide"])
            terms.append(slide_score)
        if cfg["max_contact_switch_rate"] is not None:
            chatter = min(1.0, rate / cfg["max_contact_switch_rate"])
            terms.append(1.0 - chatter)
        if cfg["max_com_dist"] is not None and dist is not None:
            com_bad = np.mean(dist > cfg["max_com_dist"])
            terms.append(1.0 - min(1.0, com_bad / cfg["max_com_ratio"]))
        if terms:
            score = max(cfg["min_weight"], float(np.mean(terms)))

    return True, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root")
    parser.add_argument("--out", required=True, help="Output YAML path")
    parser.add_argument("--pattern", default="*_jpos.npz", help="Filename pattern")
    parser.add_argument("--mjcf", default="", help="MJCF for joint limits")
    parser.add_argument("--urdf", default="", help="URDF for joint limits")
    parser.add_argument("--stage", default="all", choices=["all", "locomotion", "extended"])
    parser.add_argument("--min_len", type=float, default=0.5)
    parser.add_argument("--max_len", type=float, default=20.0)
    parser.add_argument("--min_root_z", type=float, default=0.2)
    parser.add_argument("--max_root_z", type=float, default=2.0)
    parser.add_argument("--min_root_ratio", type=float, default=0.6)
    parser.add_argument("--max_low_root_ratio", type=float, default=0.2)
    parser.add_argument("--max_root_roll", type=float, default=60.0)
    parser.add_argument("--max_root_pitch", type=float, default=60.0)
    parser.add_argument("--max_root_tilt_ratio", type=float, default=0.2)
    parser.add_argument("--allow_ground_motion", action="store_true")
    parser.add_argument("--max_root_speed", type=float, default=8.0)
    parser.add_argument("--max_root_ang_speed", type=float, default=15.0)
    parser.add_argument("--max_dof_vel", type=float, default=10.0)
    parser.add_argument("--max_dof_vel_ratio", type=float, default=0.01)
    parser.add_argument("--max_dof_acc", type=float, default=80.0)
    parser.add_argument("--max_dof_acc_ratio", type=float, default=0.01)
    parser.add_argument("--joint_limit_margin_deg", type=float, default=3.0)
    parser.add_argument("--joint_limit_ratio", type=float, default=0.01)
    parser.add_argument("--contact_height", type=float, default=0.06)
    parser.add_argument("--contact_vz", type=float, default=0.2)
    parser.add_argument("--max_foot_slide", type=float, default=0.08)
    parser.add_argument("--min_contact_ratio", type=float, default=0.1)
    parser.add_argument("--max_contact_switch_rate", type=float, default=6.0)
    parser.add_argument("--max_com_dist", type=float, default=0.25)
    parser.add_argument("--max_com_ratio", type=float, default=0.3)
    parser.add_argument("--stab_eps", type=float, default=None)
    parser.add_argument("--max_unstable_len", type=int, default=100)
    parser.add_argument("--weight_by_score", action="store_true")
    parser.add_argument("--min_weight", type=float, default=0.2)
    parser.add_argument("--require_z_up", action="store_true")
    parser.add_argument("--min_up_dot", type=float, default=0.2)
    parser.add_argument("--max_bad_up_ratio", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=0, help="Stop after N files")
    parser.add_argument("--foot_names", nargs="+", default=[
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "left_foot_link",
        "right_foot_link",
    ])
    args = parser.parse_args()

    stage_min_speed = None
    stage_max_speed = None
    if args.stage == "locomotion":
        stage_min_speed = 0.2
        stage_max_speed = 2.0
    elif args.stage == "extended":
        stage_min_speed = 0.2
        stage_max_speed = 4.0

    limits = load_joint_limits(args.mjcf or None, args.urdf or None)

    cfg = {
        "min_len": args.min_len,
        "max_len": args.max_len,
        "min_root_z": args.min_root_z,
        "max_root_z": args.max_root_z,
        "min_root_ratio": args.min_root_ratio,
        "max_low_root_ratio": args.max_low_root_ratio,
        "max_root_roll": args.max_root_roll,
        "max_root_pitch": args.max_root_pitch,
        "max_root_tilt_ratio": args.max_root_tilt_ratio,
        "allow_ground_motion": args.allow_ground_motion,
        "max_root_speed": args.max_root_speed,
        "max_root_ang_speed": args.max_root_ang_speed,
        "max_dof_vel": args.max_dof_vel,
        "max_dof_vel_ratio": args.max_dof_vel_ratio,
        "max_dof_acc": args.max_dof_acc,
        "max_dof_acc_ratio": args.max_dof_acc_ratio,
        "joint_limit_margin_deg": args.joint_limit_margin_deg,
        "joint_limit_ratio": args.joint_limit_ratio,
        "contact_height": args.contact_height,
        "contact_vz": args.contact_vz,
        "max_foot_slide": args.max_foot_slide,
        "min_contact_ratio": args.min_contact_ratio,
        "max_contact_switch_rate": args.max_contact_switch_rate,
        "max_com_dist": args.max_com_dist,
        "max_com_ratio": args.max_com_ratio,
        "stab_eps": args.stab_eps,
        "max_unstable_len": args.max_unstable_len,
        "weight_by_score": args.weight_by_score,
        "min_weight": args.min_weight,
        "foot_names": args.foot_names,
        "require_z_up": args.require_z_up,
        "min_up_dot": args.min_up_dot,
        "max_bad_up_ratio": args.max_bad_up_ratio,
        "stage_min_speed": stage_min_speed,
        "stage_max_speed": stage_max_speed,
    }

    motions = []
    count = 0
    for path in iter_npz_files(args.root, args.pattern):
        meta = load_meta(path)
        keep, score = passes_filters(meta, limits, cfg)
        if keep:
            weight = float(score) if args.weight_by_score else 1.0
            motions.append({"file": path, "weight": weight})
            count += 1
            if args.limit and count >= args.limit:
                break

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w") as f:
        yaml.safe_dump({"motions": motions}, f, sort_keys=False)

    print(f"Saved {len(motions)} motions to {args.out}")


if __name__ == "__main__":
    main()
