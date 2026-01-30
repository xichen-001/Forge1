#!/usr/bin/env python3
import argparse
import os
import sys
import xml.etree.ElementTree as ET
import yaml


def _to_float(val, default=None):
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def load_mjcf_joints(mjcf_path):
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    joints = []
    for j in root.findall(".//joint"):
        name = j.get("name")
        if not name:
            continue
        rng = j.get("range")
        rng_vals = None
        if rng:
            parts = [p for p in rng.split() if p.strip()]
            if len(parts) == 2:
                rng_vals = [float(parts[0]), float(parts[1])]
        joints.append({
            "name": name,
            "type": j.get("type"),
            "axis": j.get("axis"),
            "range": rng_vals,
            "actuatorfrcrange": j.get("actuatorfrcrange"),
            "stiffness": _to_float(j.get("stiffness"), 0.0),
            "damping": _to_float(j.get("damping"), 0.0),
            "armature": _to_float(j.get("armature"), 0.0),
            "frictionloss": _to_float(j.get("frictionloss"), None),
        })
    return joints


def check_env_configs(env_paths, mjcf_path):
    mismatches = []
    for p in env_paths:
        if not p:
            continue
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        char_file = cfg.get("char_file")
        if not char_file:
            mismatches.append((p, "char_file missing"))
            continue
        norm_char = os.path.normpath(char_file)
        norm_mjcf = os.path.normpath(mjcf_path)
        if norm_char != norm_mjcf:
            mismatches.append((p, f"char_file={char_file} != {mjcf_path}"))
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", required=True, help="Path to g1.xml")
    parser.add_argument("--out", required=True, help="Output yaml for joint params")
    parser.add_argument("--env_config", action="append", default=[],
                        help="Optional env config(s) to verify char_file")
    args = parser.parse_args()

    joints = load_mjcf_joints(args.mjcf)
    out = {
        "mjcf": os.path.abspath(args.mjcf),
        "num_joints": len(joints),
        "joints": joints,
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    mismatches = check_env_configs(args.env_config, args.mjcf)
    if mismatches:
        print("Env config mismatches:")
        for p, msg in mismatches:
            print(f"- {p}: {msg}")
        sys.exit(2)

    print(f"Wrote joint params to {args.out} ({len(joints)} joints)")


if __name__ == "__main__":
    main()
