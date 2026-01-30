# scripts/render_objs_pyrender.py
import argparse
import os

import numpy as np
import pyrender
import trimesh
from PIL import Image


def render_obj_sequence(obj_dir, out_dir, width=640, height=480):
    files = sorted([f for f in os.listdir(obj_dir) if f.endswith(".obj")])
    os.makedirs(out_dir, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    # default camera pose (tweak if needed)
    cam_pose = np.array([[1, 0, 0, 0], [0, 1, 0, -2.0], [0, 0, 1, 1.2], [0, 0, 0, 1]])
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    for i, fn in enumerate(files):
        path = os.path.join(obj_dir, fn)
        mesh = trimesh.load(path, process=False)
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])
        pm = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(pm)
        scene.add(light, pose=np.eye(4))
        scene.add(camera, pose=cam_pose)
        color, depth = renderer.render(scene)
        im = Image.fromarray(color)
        out_path = os.path.join(out_dir, f"frame{i:04d}.png")
        im.save(out_path)
        print("Saved", out_path)
    renderer.delete()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--w", type=int, default=640)
    parser.add_argument("--h", type=int, default=480)
    args = parser.parse_args()
    render_obj_sequence(args.obj_dir, args.out_dir, args.w, args.h)
