import argparse
import os
import subprocess
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "mimickit"))


def _load_configs(args):
    env_file = ""
    engine_file = ""
    agent_file = ""

    if args.arg_file:
        from util import arg_parser
        ap = arg_parser.ArgParser()
        ap.load_file(args.arg_file)
        env_file = ap.parse_string("env_config", env_file)
        engine_file = ap.parse_string("engine_config", engine_file)
        agent_file = ap.parse_string("agent_config", agent_file)

    if args.env_config:
        env_file = args.env_config
    if args.engine_config:
        engine_file = args.engine_config
    if args.agent_config:
        agent_file = args.agent_config

    if not env_file or not engine_file or not agent_file:
        raise ValueError("env_config, engine_config, agent_config must be set (directly or via --arg_file)")

    with open(env_file, "r") as f:
        env_config = yaml.safe_load(f)

    return env_file, engine_file, agent_file, env_config


def _build_camera(env, env_config):
    from util import camera
    cam_pos = np.array([0.0, -5.0, 3.0])
    cam_target = np.array([0.0, 0.0, 0.0])
    cam_mode = env_config.get("camera_mode", "still")
    cam_mode = camera.CameraMode[cam_mode]
    return camera.Camera(mode=cam_mode, engine=env._engine, pos=cam_pos, target=cam_target)


def _open_ffmpeg(out_path, width, height, fps):
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        out_path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arg_file", default="", help="MimicKit args file")
    parser.add_argument("--env_config", default="", help="Env config yaml")
    parser.add_argument("--engine_config", default="", help="Engine config yaml (use camera-enabled)")
    parser.add_argument("--agent_config", default="", help="Agent config yaml")
    parser.add_argument("--model_file", default="", help="Policy checkpoint")
    parser.add_argument("--out", default="output/rollout.mp4", help="Output mp4 path")
    parser.add_argument("--num_frames", type=int, default=900, help="Number of frames to record")
    parser.add_argument("--fps", type=int, default=30, help="Video fps")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (use 1 for video)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    env_file, engine_file, agent_file, env_config = _load_configs(args)
    with open(engine_file, "r") as f:
        engine_cfg = yaml.safe_load(f)
    engine_name = engine_cfg.get("engine_name", "")

    if engine_name == "isaac_gym":
        import torch
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            del sys.modules["torch"]
        import isaacgym.gymapi  # noqa: F401
        if torch_mod is not None:
            sys.modules["torch"] = torch_mod
    else:
        import torch

    import util.mp_util as mp_util
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder

    device = args.device
    mp_util.init(0, 1, device, 0)

    env = env_builder.build_env(env_file, engine_file, args.num_envs, device, visualize=False)
    agent = agent_builder.build_agent(agent_file, env, device)

    if args.model_file:
        agent.load(args.model_file)
    agent.eval()

    cam = _build_camera(env, env_config)
    width = int(env._engine._camera_width)
    height = int(env._engine._camera_height)
    proc = _open_ffmpeg(args.out, width, height, args.fps)

    obs, info = env.reset()
    with torch.no_grad():
        for _ in range(args.num_frames):
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)

            cam.update()
            frame = env._engine.capture_rgb()
            frame = frame[..., :3]  # RGBA -> RGB
            proc.stdin.write(frame.tobytes())

            if torch.any(done != 0):
                obs, info = env.reset()

    proc.stdin.close()
    proc.wait()
    print(f"Saved video to {args.out}")


if __name__ == "__main__":
    main()
