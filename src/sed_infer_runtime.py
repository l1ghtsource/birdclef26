from __future__ import annotations

import numpy as np
import torch
import torchaudio
from torchvision.transforms import v2

INFER_S = 5.0
NUM_SEGMENTS = 12


def parse_row_id(row_id: str) -> tuple[str, int]:
    stem, end = str(row_id).rsplit("_", 1)
    return stem, int(end)


def mel_waveform_to_tensor(wave_1d: np.ndarray, *, sr: int, target_sr: int) -> torch.Tensor:
    x = torch.from_numpy(wave_1d.astype(np.float32, copy=False))
    if x.dim() == 1:
        x = x.unsqueeze(0)
    elif x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    if sr != target_sr:
        x = torchaudio.functional.resample(x, sr, target_sr)
    return x


def compute_deltas(specgram: torch.Tensor, win_length: int = 5, mode: str = "replicate") -> torch.Tensor:
    device = specgram.device
    dtype = specgram.dtype
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])
    assert win_length >= 3
    n = (win_length - 1) // 2
    denom = n * (n + 1) * (2 * n + 1) / 3
    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)
    kernel = torch.arange(-n, n + 1, 1, device=device, dtype=dtype).repeat(specgram.shape[1], 1, 1)
    output = torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    return output.reshape(shape)


def normalize_melspec_tensor(x: torch.Tensor, p: dict) -> torch.Tensor:
    method = p["norm_method"]
    top_db = p["mel_top_db"]
    if method == "none":
        return x
    if method == "db_scaled":
        return torch.clamp((x + top_db) / top_db, 0.0, 1.0)
    if method == "per_sample_minmax":
        lo, hi = x.min(), x.max()
        return (x - lo) / (hi - lo).clamp_min(1e-8)
    if method == "per_sample_absmax":
        return x / x.abs().max().clamp_min(1e-8)
    if method == "per_sample_std":
        return (x - x.mean()) / x.std().clamp_min(1e-6)
    raise NotImplementedError(method)


class MelRuntime:
    def __init__(self, config: dict):
        self.config = config
        mel = config["mel_spec_params"]
        self.sr = int(mel["sample_rate"])
        self.chunk_s = float(config["dataset"]["chunk_duration_s"])
        self.infer_s = INFER_S
        self.k = int(round(self.chunk_s / self.infer_s))
        self.h_mel, self.w_mel = mel["mel_image_size"]
        self.do_resize = True
        self.mel_cpu = torchaudio.transforms.MelSpectrogram(
            sample_rate=mel["sample_rate"],
            n_fft=mel["n_fft"],
            hop_length=mel["hop_length"],
            n_mels=mel["n_mels"],
            f_min=mel["f_min"],
            f_max=mel["f_max"],
        )
        self.db_cpu = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=mel["mel_top_db"])
        self.resize = v2.Resize(size=(self.h_mel, self.w_mel))

    def audio_to_spec_infer(self, wave_1d: np.ndarray, sr_file: int) -> torch.Tensor:
        p = self.config["mel_spec_params"]
        x = mel_waveform_to_tensor(wave_1d, sr=sr_file, target_sr=p["sample_rate"])
        with torch.no_grad():
            mel = self.mel_cpu(x)
            if p.get("use_channel_agnostic_atodb", False):
                amin = 1e-10
                ref_value = 1.0
                db_mult = np.log10(max(amin, ref_value))
                spec = torchaudio.functional.amplitude_to_DB(
                    mel,
                    multiplier=10.0,
                    amin=amin,
                    db_multiplier=db_mult,
                    top_db=p["mel_top_db"],
                )
            else:
                spec = self.db_cpu(mel)
            spec = normalize_melspec_tensor(spec, p)
            if p["mel_delta_stack"]:
                delta_1 = compute_deltas(spec)
                delta_2 = compute_deltas(delta_1)
                spec = torch.cat([spec, delta_1, delta_2], dim=0)
            else:
                nc = int(self.config["model"].get("num_channels", 3))
                if nc == 1:
                    pass
                elif nc == 3:
                    spec = torch.cat([spec, spec, spec], dim=0)
                else:
                    raise NotImplementedError(f"Unsupported model.num_channels={nc}")
            if self.do_resize:
                spec = self.resize(spec)
        return spec.float()

    def prepare_mel_batch(self, wave_np: np.ndarray, sr_native: int) -> torch.Tensor:
        if self.config["dataset"].get("do_peak_norm", False):
            m = np.max(np.abs(wave_np))
            if m >= 1e-8:
                wave_np = wave_np / m

        x = mel_waveform_to_tensor(wave_np, sr=sr_native, target_sr=self.sr)
        w = x.squeeze(0).detach().cpu().numpy().astype(np.float32)
        total_samples = int(60 * self.sr)
        if w.shape[0] < total_samples:
            w = np.pad(w, (0, total_samples - w.shape[0]), mode="constant")
        else:
            w = w[:total_samples]

        block_samples = int(round(self.chunk_s * self.sr))
        num_blocks = int(np.ceil(total_samples / block_samples))
        padded_len = num_blocks * block_samples
        if w.shape[0] < padded_len:
            w = np.pad(w, (0, padded_len - w.shape[0]), mode="constant")

        tiles = []
        for b in range(num_blocks):
            start = b * block_samples
            end = start + block_samples
            tiles.append(self.audio_to_spec_infer(w[start:end], self.sr))
        return torch.stack(tiles, dim=0).float()


def seglog_to_12_segment_probs(seglog: np.ndarray, *, k: int, num_blocks: int) -> np.ndarray:
    """Map (num_blocks, T, C) segment logits to (12, C) probabilities."""
    num_classes = seglog.shape[2]
    t_out = seglog.shape[1]
    t_per_seg = t_out // k
    seg_probs = np.zeros((NUM_SEGMENTS, num_classes), dtype=np.float32)
    for b in range(num_blocks):
        for i in range(k):
            global_idx = b * k + i
            if global_idx >= NUM_SEGMENTS:
                break
            start_t = i * t_per_seg
            end_t = t_out if i == k - 1 else start_t + t_per_seg
            seg_l = seglog[b, start_t:end_t, :]
            seg_max = 1.0 / (1.0 + np.exp(-seg_l.max(axis=0)))
            seg_probs[global_idx] = seg_max
    return seg_probs


def infer_stem_probs(
    model: torch.nn.Module,
    mel_runtime: MelRuntime,
    wave: np.ndarray,
    sr: int,
    *,
    device: torch.device,
    col_to_model: np.ndarray,
    row_indices: list[tuple[int, float]],
    out_probs: np.ndarray,
) -> None:
    mel_b = mel_runtime.prepare_mel_batch(wave, sr)
    block_samples = int(round(mel_runtime.chunk_s * mel_runtime.sr))
    total_samples_60s = int(60 * mel_runtime.sr)
    num_blocks = int(np.ceil(total_samples_60s / block_samples))

    with torch.no_grad():
        out = model(mel_b.to(device))
        seglog = out["segmentwise_logit"].detach().float().cpu().numpy()

    seg_probs = seglog_to_12_segment_probs(seglog, k=mel_runtime.k, num_blocks=num_blocks)
    for row_idx, end_sec in row_indices:
        si = int(end_sec // mel_runtime.infer_s) - 1
        si = max(0, min(NUM_SEGMENTS - 1, si))
        out_probs[row_idx] = seg_probs[si][col_to_model]
