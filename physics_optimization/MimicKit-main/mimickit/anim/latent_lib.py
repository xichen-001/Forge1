import os
import numpy as np
import torch


class LatentLib:
    def __init__(self, latent_file, device, fps=30):
        if not os.path.exists(latent_file):
            raise FileNotFoundError(f"latent_file not found: {latent_file}")

        self._device = device
        latent, lengths, file_fps = self._load_latents(latent_file)

        if file_fps is not None:
            fps = file_fps

        self._fps = float(fps)
        self._dt = 1.0 / self._fps

        # latent: (N, T, D)
        self._latent = torch.tensor(latent, dtype=torch.float32, device=self._device)
        self._lengths = torch.tensor(lengths, dtype=torch.long, device=self._device)

        self._num_motions = self._latent.shape[0]
        self._motion_weights = torch.ones(self._num_motions, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

    def get_num_motions(self):
        return self._num_motions

    def get_latent_dim(self):
        return self._latent.shape[-1]

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._get_motion_len_sec(motion_ids)

        if truncate_time is not None:
            motion_len = torch.clamp(motion_len - truncate_time, min=0.0)

        motion_time = phase * motion_len
        return motion_time

    def get_latent_frame(self, motion_ids, motion_times):
        # motion_times: seconds
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)

        z0 = self._latent[motion_ids, frame_idx0]
        z1 = self._latent[motion_ids, frame_idx1]
        blend = blend.unsqueeze(-1)
        z = (1.0 - blend) * z0 + blend * z1
        return z

    def get_latent_window(self, motion_ids, motion_times, window=1):
        if window <= 1:
            z = self.get_latent_frame(motion_ids, motion_times)
            return z.unsqueeze(1)

        # past window ending at current time
        time_steps = -self._dt * torch.arange(window - 1, -1, -1, device=self._device)
        times = motion_times.unsqueeze(-1) + time_steps.unsqueeze(0)

        flat_times = times.reshape(-1)
        flat_ids = motion_ids.unsqueeze(-1).expand(-1, window).reshape(-1)

        z = self.get_latent_frame(flat_ids, flat_times)
        z = z.view(motion_ids.shape[0], window, -1)
        return z

    def _get_motion_len_sec(self, motion_ids):
        lengths = self._lengths[motion_ids].float()
        motion_len = (lengths - 1.0) / self._fps
        return torch.clamp(motion_len, min=0.0)

    def _calc_frame_blend(self, motion_ids, motion_times):
        lengths = self._lengths[motion_ids].float()
        max_frame = torch.clamp(lengths - 1.0, min=0.0)

        frame_float = motion_times * self._fps
        frame_float = torch.min(torch.max(frame_float, torch.zeros_like(max_frame)), max_frame)

        frame_idx0 = torch.floor(frame_float).long()
        frame_idx1 = torch.min(frame_idx0 + 1, max_frame.long())
        blend = frame_float - frame_idx0.float()
        return frame_idx0, frame_idx1, blend

    def _load_latents(self, latent_file):
        # Expect .npz with latent/lengths/(optional)fps
        if latent_file.endswith(".npz"):
            data = np.load(latent_file, allow_pickle=True)
            latent = data["latent"]
            lengths = data["lengths"]
            file_fps = None
            if "fps" in data:
                file_fps = float(data["fps"])
            return latent, lengths, file_fps

        raise ValueError("latent_file must be .npz with keys: latent, lengths, (optional) fps")
