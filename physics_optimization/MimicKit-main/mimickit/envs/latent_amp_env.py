import torch

import envs.amp_env as amp_env
import anim.latent_lib as latent_lib


class LatentAMPEnv(amp_env.AMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize):
        self._latent_file = env_config["latent_file"]
        self._latent_fps = env_config.get("latent_fps", 30)
        self._latent_window = int(env_config.get("latent_window", 1))
        self._latent_mode = env_config.get("latent_mode", "sync")

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize)

        if self._latent_mode != "sync":
            raise ValueError("latent_mode must be 'sync' for now")
        return
    
    def _build_data_buffers(self):
        if not hasattr(self, "_latent_lib"):
            self._latent_lib = latent_lib.LatentLib(
                latent_file=self._latent_file,
                device=self._device,
                fps=self._latent_fps,
            )
        super()._build_data_buffers()
        return

    def _compute_obs(self, env_ids=None):
        if not hasattr(self, "_latent_lib"):
            raise RuntimeError("latent_lib is not initialized")
        obs = super()._compute_obs(env_ids)

        if env_ids is None:
            motion_ids = self._motion_ids
            motion_times = self._get_motion_times()
        else:
            motion_ids = self._motion_ids[env_ids]
            motion_times = self._get_motion_times(env_ids)

        num_latents = self._latent_lib.get_num_motions()
        if num_latents <= 0:
            raise RuntimeError("latent_lib has no motions loaded")
        if num_latents == 1:
            motion_ids = torch.zeros_like(motion_ids)
        else:
            motion_ids = torch.remainder(motion_ids, num_latents)

        latent = self._latent_lib.get_latent_window(
            motion_ids=motion_ids,
            motion_times=motion_times,
            window=self._latent_window,
        )
        latent = latent.reshape(latent.shape[0], -1)

        obs = torch.cat([obs, latent], dim=-1)
        return obs
