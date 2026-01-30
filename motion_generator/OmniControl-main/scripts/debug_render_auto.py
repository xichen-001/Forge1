#!/usr/bin/env python3
# scripts/debug_render_auto.py
# Usage:
# python scripts/debug_render_auto.py --obj path/to/frame000.obj --out /home/yxc/debug_render_frame000.png

import argparse
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image


def look_at(eye, target, up):
    z = eye - target
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        # fallback: choose arbitrary orthogonal vector
        if abs(up[0]) < 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        else:
            tmp = np.array([0.0, 1.0, 0.0])
        x = np.cross(tmp, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[0:3, 0] = x
    M[0:3, 1] = y
    M[0:3, 2] = z
    M[0:3, 3] = eye
    return M


parser = argparse.ArgumentParser()
parser.add_argument("--obj", required=True)
parser.add_argument("--out", default="/home/yxc/debug_render_frame000.png")
parser.add_argument("--w", type=int, default=1024)
parser.add_argument("--h", type=int, default=1024)
parser.add_argument("--show_stats", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.obj):
    raise RuntimeError("obj not found: " + args.obj)

m = trimesh.load(args.obj, process=False)
if args.show_stats:
    print("verts/faces:", len(m.vertices), len(m.faces))
    print("bounds:", m.bounds, "centroid:", m.centroid, "extents:", m.extents)
centroid = m.centroid
ext = m.extents
# detect up axis as the axis with largest extent
up_idx = int(np.argmax(ext))
axes = {0: np.array([1.0, 0.0, 0.0]), 1: np.array([0.0, 1.0, 0.0]), 2: np.array([0.0, 0.0, 1.0])}
up = axes[up_idx]
# choose a viewing axis that's not the up axis: pick the axis with second largest extent
other_idx = int(np.argsort(ext)[-2])
view_axis = {0: np.array([0.0, 0.0, 1.0]), 1: np.array([0.0, 0.0, 1.0]), 2: np.array([0.0, 1.0, 0.0])}[other_idx]
# We'll place camera along view_axis direction relative to centroid
# distance relative to the model size
dist = max(ext) * 3.0 if max(ext) > 0 else 2.0
eye = centroid + view_axis * dist
# create scene with strong ambient and directional lights
scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[1.0, 1.0, 1.0])
pm = pyrender.Mesh.from_trimesh(m, smooth=False)
scene.add(pm)

# camera
cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
cam_pose = look_at(eye, centroid, up)
scene.add(cam, pose=cam_pose)

# add multiple directional lights around the camera
light_positions = [
    eye,  # from camera position
    centroid + view_axis * (-dist * 0.3) + up * dist * 0.5,
    centroid + np.array([dist * 0.6, 0.0, 0.0]),  # side light
]
for lp in light_positions:
    lp_pose = look_at(lp, centroid, up)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=60.0), pose=lp_pose)

# render
r = pyrender.OffscreenRenderer(viewport_width=args.w, viewport_height=args.h)
color, depth = r.render(scene)
r.delete()
im = Image.fromarray(color)
im.save(args.out)
print("Saved debug image to", args.out)
