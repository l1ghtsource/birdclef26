from pathlib import Path

import torch
import torchaudio

from src.create_folds import n_train_audio_rows
from src.dataset import (
    PaddingMode,
    build_soundscape_items,
    load_audio_soundfile,
    pad_waveform_to_length,
    reshape_waveform,
)
from src.melspec import AudioToSpec
from src.taxonomy_merge import build_merged_label_to_idx


def encode_labels(primary, secondary, label_to_idx, secondary_w):
    y = torch.zeros(len(label_to_idx), dtype=torch.float32)
    for p in primary:
        if p in label_to_idx:
            y[label_to_idx[p]] = 1.0
    for s in secondary:
        if s in label_to_idx:
            i = label_to_idx[s]
            y[i] = max(float(y[i]), secondary_w)
    return y


def load_ss_wave_bank(item, sample_rate, chunk_samples, rng, pad_mode):
    offset = round(float(item.t0_sec or 0.0) * sample_rate)
    wav, sr = load_audio_soundfile(item.path, frame_offset=offset, num_frames=chunk_samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    c = chunk_samples
    t = wav.shape[1]
    if t < c:
        wav = pad_waveform_to_length(wav, c, pad_mode, rng, train=False)
    elif t > c:
        wav = wav[:, :c]
    return wav


def build_soundscape_augmentation_bank(cfg):
    if not cfg["dataset"]["use_train_soundscapes"]:
        return None, None

    data_root = Path(cfg["data_root"])
    ds = cfg["dataset"]
    oa = cfg["online_aug"]
    wave_level = oa["wave_level"]
    use_rs = bool(ds["use_reshape_waves_not_melspec"])
    wave_w = int(ds["wave_reshape_width"])

    label_to_idx = build_merged_label_to_idx(data_root, ds)

    items = build_soundscape_items(
        data_root,
        data_root / ds["soundscapes_labels_csv"],
        ds["train_soundscapes_subdir"],
        label_to_idx,
        float(ds["chunk_duration_s"]),
        float(ds["soundscape_label_bin_s"]),
    )
    if not items:
        return None, None

    n_audio = n_train_audio_rows(data_root, ds)
    train_idx = cfg.get("ss_bank_train_indices")
    allowed: set[int] | None = None
    if train_idx is not None:
        allowed = {int(i) for i in train_idx}
    elif float(oa.get("ss_bank_share", 0.0)) > 0.0:
        raise ValueError(
            "use_ss_bank with ss_bank_share>0 requires cfg['ss_bank_train_indices'] "
            "(global indices into BirdClefTrainingDataset for the current fold's train Subset). "
            "Pass list(train_ds.indices) from scripts/train.py or scripts/pretrain.py."
        )

    sample_rate = int(ds["sample_rate"])
    chunk_samples = round(float(ds["chunk_duration_s"]) * sample_rate)
    pad_mode = PaddingMode(ds["padding_mode"])
    rng = torch.Generator().manual_seed(int(cfg["seed"]) + 1337)

    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    secondary_w = float(ds["secondary_label_weight"])

    audio_to_spec = AudioToSpec(cfg) if not wave_level else None

    for j, item in enumerate(items):
        global_idx = n_audio + j
        if allowed is not None and global_idx not in allowed:
            continue
        wav = load_ss_wave_bank(item, sample_rate, chunk_samples, rng, pad_mode)
        y = encode_labels(item.primary_labels, item.secondary_labels, label_to_idx, secondary_w)

        if wave_level:
            x = wav
        elif use_rs:
            x = reshape_waveform(wav, wave_w)
        else:
            x = audio_to_spec(wav, train=False)

        xs.append(x.cpu())
        ys.append(y.cpu())

    if not xs:
        return None, None

    bx = torch.stack(xs, dim=0)
    by = torch.stack(ys, dim=0)

    return bx, by
