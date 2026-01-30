#!/usr/bin/env python3
# scripts/render_with_ground_fix.py
import argparse
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image


def normalize(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def look_at(eye, target, up):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        tmp = np.array([1.0, 0.0, 0.0])
        x = np.cross(tmp, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[0:3, 0] = x
    M[0:3, 1] = y
    M[0:3, 2] = z
    M[0:3, 3] = eye
    return M


def make_ground(centroid, floor_y, plane_size=5.0, thickness=0.01, up_idx=1):
    size = max(plane_size, 2.0)
    if up_idx == 1:
        extents = (size, thickness, size)
        center = np.array([centroid[0], floor_y - thickness / 2.0, centroid[2]])
        transform = np.eye(4)
        transform[0:3, 3] = center
    elif up_idx == 2:
        extents = (size, size, thickness)
        center = np.array([centroid[0], centroid[1], floor_y - thickness / 2.0])
        transform = np.eye(4)
        transform[0:3, 3] = center
    else:
        extents = (thickness, size, size)
        center = np.array([floor_y - thickness / 2.0, centroid[1], centroid[2]])
        transform = np.eye(4)
        transform[0:3, 3] = center
    ground = trimesh.creation.box(extents=extents, transform=transform)
    return ground


def render_sequence(obj_dir, out_dir, w=1024, h=1024):
    files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".obj")])
    assert len(files) > 0, "no obj files"
    os.makedirs(out_dir, exist_ok=True)

    first = trimesh.load(os.path.join(obj_dir, files[0]), process=False)
    bounds = first.bounds
    centroid = first.centroid
    ext = first.extents
    up_idx = int(np.argmax(ext))
    print("Detected up idx:", up_idx, "centroid:", centroid, "extents:", ext)

    # choose view and camera
    if up_idx == 1:
        view_dir = np.array([0.8, -1.0, 0.45])
    elif up_idx == 2:
        view_dir = np.array([0.8, -1.0, 0.45])
    else:
        view_dir = np.array([-1.0, 0.8, 0.45])
    view_dir = normalize(view_dir)
    dist = max(ext) * 2.2  # 更靠近一点 (之前是3.0)
    eye = centroid + view_dir * dist
    up_vec = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0]), 2: np.array([0.0, 0.0, 1.0])}[up_idx]
    floor_y = float(bounds[0, up_idx])

    for i, fn in enumerate(files):
        print("Rendering", i, fn)
        path = os.path.join(obj_dir, fn)
        m = trimesh.load(path, process=False)
        scene = pyrender.Scene(bg_color=[0.12, 0.12, 0.12, 1.0], ambient_light=[0.12, 0.12, 0.12])

        # material: darker orange
        try:
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(0.78, 0.45, 0.22, 1.0), metallicFactor=0.0, roughnessFactor=0.6
            )
        except Exception:
            material = None

        try:
            pm = pyrender.Mesh.from_trimesh(m, smooth=True, material=material)
        except Exception:
            pm = pyrender.Mesh.from_trimesh(m, smooth=False)
        scene.add(pm)

        # ground: dark grey
        ground_tr = make_ground(centroid, floor_y, plane_size=max(ext) * 6, thickness=0.01, up_idx=up_idx)
        try:
            ground_mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(0.35, 0.35, 0.35, 1.0), metallicFactor=0.0, roughnessFactor=1.0
            )
            pm_ground = pyrender.Mesh.from_trimesh(ground_tr, material=ground_mat)
        except Exception:
            pm_ground = pyrender.Mesh.from_trimesh(ground_tr)
        scene.add(pm_ground)

        # camera
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        cam_pose = look_at(eye, centroid, up_vec)
        scene.add(cam, pose=cam_pose)

        # lights (much lower intensities than before)
        main_light = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)  # was 80
        fill_light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.5)  # was 20
        back_light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)  # was 10

        scene.add(main_light, pose=cam_pose)
        fill_pos = centroid + normalize(np.array([-0.6, -0.8, 1.0])) * dist
        scene.add(fill_light, pose=look_at(fill_pos, centroid, up_vec))
        back_pos = centroid + normalize(np.array([0.8, 0.8, 0.6])) * dist
        scene.add(back_light, pose=look_at(back_pos, centroid, up_vec))

        r = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
        color, depth = r.render(scene)
        r.delete()

        # clamp values and save
        color = np.clip(color, 0, 255).astype("uint8")
        im = Image.fromarray(color)
        out_path = os.path.join(out_dir, f"frame{i:04d}.png")
        im.save(out_path)
    print("Done, saved frames to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--w", type=int, default=1024)
    parser.add_argument("--h", type=int, default=1024)
    args = parser.parse_args()
    render_sequence(args.obj_dir, args.out_dir, args.w, args.h)
