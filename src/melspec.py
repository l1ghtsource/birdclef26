import math

import cv2
import librosa
import torch
import torchaudio
import torchaudio.functional as taF
import torchvision.transforms.v2 as v2

from src.offline_augmentations import (
    AddGaussianNoise,
    FrequencyShift,
    LocalGlobalStretch,
    SpecAugment,
    TimeShift,
    apply_random_gain_db,
    filt_aug,
)


def compute_deltas(specgram, win_length=5, mode="replicate"):
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
    output = output.reshape(shape)
    return output


class AudioToSpec(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.aug_config = config["mel_spec_aug"]
        self.mel_spec_params = config["mel_spec_params"]
        self.use_top4_method = bool(self.mel_spec_params.get("use_top4_method", False))
        self.top4_preset = self._resolve_top4_preset(config)
        self.torchaudio_norm = bool(self.mel_spec_params.get("torchaudio_norm", False))
        self.use_traceable_melspec = bool(self.mel_spec_params.get("use_traceable_melspec", False))
        self.use_channel_agnostic_atodb = bool(self.mel_spec_params.get("use_channel_agnostic_atodb", False))
        self.model_num_channels = int(config.get("model", {}).get("num_channels", 3))
        msp = self.mel_spec_params
        if self.use_top4_method:
            # top4 path uses librosa melspectrogram + power_to_db(ref=1.0)
            self.mel_frontend = None
        elif self.use_traceable_melspec:
            from src.traceable_melspec import TraceableMelspec

            self.mel_frontend = TraceableMelspec(
                win_length=msp.get("win_length"),
                hop_length=msp["hop_length"],
                power=float(msp.get("power", 2.0)),
                normalized=self.torchaudio_norm,
                center=bool(msp.get("stft_center", True)),
                pad_mode=str(msp.get("stft_pad_mode", "reflect")),
                n_mels=msp["n_mels"],
                sample_rate=msp["sample_rate"],
                f_min=float(msp["f_min"]),
                f_max=msp.get("f_max"),
                n_fft=msp["n_fft"],
                norm=msp.get("mel_norm"),
                mel_scale=str(msp.get("mel_scale", "htk")),
                trainable=False,
                quantizable=False,
                stft_verbose=bool(msp.get("nnaudio_stft_verbose", False)),
            )
        else:
            self.mel_frontend = torchaudio.transforms.MelSpectrogram(
                sample_rate=msp["sample_rate"],
                n_fft=msp["n_fft"],
                hop_length=msp["hop_length"],
                n_mels=msp["n_mels"],
                f_min=msp["f_min"],
                f_max=msp["f_max"],
                normalized=self.torchaudio_norm,
            )
        if self.use_top4_method:
            self.db_transform = None
        elif not self.use_channel_agnostic_atodb:
            self.db_transform = torchaudio.transforms.AmplitudeToDB(
                stype="power", top_db=self.mel_spec_params["mel_top_db"]
            )
        else:
            self.db_transform = None

        image_size = self.mel_spec_params["mel_image_size"]

        if image_size is None:
            freq_mask_param = 10
            time_mask_param = 10
        else:
            freq_mask_param = int(image_size[0] * 0.04)
            time_mask_param = int(image_size[1] * 0.1)

        self.spec_aug = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            num_freq_masks=self.aug_config["spec_aug_num_freq_masks"],
            num_time_masks=self.aug_config["spec_aug_num_time_masks"],
            p=self.aug_config["p_spec_aug"],
        )
        self.stretch_aug = LocalGlobalStretch(
            global_stretch_prob=self.aug_config["stretch_global_prob"],
            local_stretch_prob=self.aug_config["stretch_local_prob"],
            max_global_stretch=self.aug_config["stretch_max_global"],
            max_local_stretch=self.aug_config["stretch_max_local"],
            max_local_regions=self.aug_config["stretch_max_local_regions"],
        )
        self.top_db = self.mel_spec_params["mel_top_db"]
        self.delta_stack = self.mel_spec_params["mel_delta_stack"]
        if self.delta_stack and self.model_num_channels != 3:
            raise ValueError(
                "mel_delta_stack=True produces 3 mel planes; set cfg.model.num_channels=3 for the backbone."
            )

        if image_size is not None:
            self.resize_transforms = v2.Compose([v2.Resize(size=image_size)])
        else:
            self.resize_transforms = None

        self.train_transforms = v2.Compose(
            [
                TimeShift(
                    max_shift_pct=self.aug_config["time_shift_max_pct"],
                    p=self.aug_config["p_time_shift"],
                ),
                FrequencyShift(
                    max_shift_pct=self.aug_config["freq_shift_max_pct"],
                    p=self.aug_config["p_freq_shift"],
                ),
                AddGaussianNoise(std=self.aug_config["gaussian_noise_std"], p=self.aug_config["p_gaussian_noise"]),
                v2.RandomErasing(
                    p=self.aug_config["p_random_erasing"],
                    scale=self.aug_config["random_erasing_scale"],
                    ratio=self.aug_config["random_erasing_ratio"],
                ),
            ]
        )

        self.norm_method = self.mel_spec_params["norm_method"]
        self.normalize_melspec_eps = float(self.mel_spec_params.get("normalize_melspec_eps", 1e-6))
        self.normalize_melspec_exportable = bool(self.mel_spec_params.get("normalize_melspec_exportable", False))
        self.top4_specaug_p = float(self.aug_config.get("top4_specaug_p", 0.5))
        self.top4_specaug_num_masks_min = int(self.aug_config.get("top4_specaug_num_masks_min", 1))
        self.top4_specaug_num_masks_max = int(self.aug_config.get("top4_specaug_num_masks_max", 3))
        self.top4_specaug_width_min = int(self.aug_config.get("top4_specaug_width_min", 5))
        self.top4_specaug_width_max = int(self.aug_config.get("top4_specaug_width_max", 20))
        self.top4_specaug_gain_min = float(self.aug_config.get("top4_specaug_gain_min", 0.8))
        self.top4_specaug_gain_max = float(self.aug_config.get("top4_specaug_gain_max", 1.2))

    def _apply_top4_specaug(self, spec: torch.Tensor) -> torch.Tensor:
        # top4-style spectrogram augmentations: time masks, freq masks, random gain.
        if torch.rand(1, device=spec.device).item() < self.top4_specaug_p:
            n_masks = int(
                torch.randint(
                    self.top4_specaug_num_masks_min,
                    self.top4_specaug_num_masks_max + 1,
                    (1,),
                    device=spec.device,
                ).item()
            )
            max_width = min(self.top4_specaug_width_max, spec.shape[2])
            min_width = min(self.top4_specaug_width_min, max_width)
            if min_width > 0 and max_width >= min_width:
                for _ in range(n_masks):
                    width = int(torch.randint(min_width, max_width + 1, (1,), device=spec.device).item())
                    start_max = max(1, spec.shape[2] - width + 1)
                    start = int(torch.randint(0, start_max, (1,), device=spec.device).item())
                    spec[0, :, start : start + width] = 0

        if torch.rand(1, device=spec.device).item() < self.top4_specaug_p:
            n_masks = int(
                torch.randint(
                    self.top4_specaug_num_masks_min,
                    self.top4_specaug_num_masks_max + 1,
                    (1,),
                    device=spec.device,
                ).item()
            )
            max_height = min(self.top4_specaug_width_max, spec.shape[1])
            min_height = min(self.top4_specaug_width_min, max_height)
            if min_height > 0 and max_height >= min_height:
                for _ in range(n_masks):
                    height = int(torch.randint(min_height, max_height + 1, (1,), device=spec.device).item())
                    start_max = max(1, spec.shape[1] - height + 1)
                    start = int(torch.randint(0, start_max, (1,), device=spec.device).item())
                    spec[0, start : start + height, :] = 0

        if torch.rand(1, device=spec.device).item() < self.top4_specaug_p:
            gain = torch.empty(1, device=spec.device, dtype=spec.dtype).uniform_(
                self.top4_specaug_gain_min,
                self.top4_specaug_gain_max,
            )
            spec = spec * gain
        return spec

    def forward(self, audio, train):
        if train:
            audio = apply_random_gain_db(
                audio,
                min_db=self.aug_config["random_gain_min_db"],
                max_db=self.aug_config["random_gain_max_db"],
                p=self.aug_config["p_random_gain_db"],
            )

        if self.use_top4_method:
            spec = self.top4_mel_frontend(audio)
        else:
            spec = self.mel_frontend(audio)

        if train:
            spec = filt_aug(
                spec,
                db_range=list(self.aug_config["filt_aug_db_range"]),
                n_band=list(self.aug_config["filt_aug_n_band"]),
                min_bw=self.aug_config["filt_aug_min_bw"],
                filter_type=self.aug_config["filt_aug_filter_type"],
                p=self.aug_config["p_filt_aug"],
            )

        if self.use_top4_method:
            pass
        elif self.use_channel_agnostic_atodb:
            amin = 1e-10
            ref_value = 1.0
            db_mult = math.log10(max(amin, ref_value))
            spec = taF.amplitude_to_DB(spec, 10.0, amin, db_mult, self.top_db)
        else:
            spec = self.db_transform(spec)

        spec = self.normalize_melspec(spec)

        if self.delta_stack:
            delta_1 = compute_deltas(spec)
            delta_2 = compute_deltas(delta_1)
            spec = torch.cat([spec, delta_1, delta_2], dim=0)
        else:
            nc = self.model_num_channels
            if nc == 1:
                pass
            elif nc == 3:
                spec = torch.cat([spec, spec, spec], dim=0)
            else:
                raise NotImplementedError(f"model.num_channels={nc} not supported (use 1 or 3).")

        if train:
            spec = self.stretch_aug(spec)
            spec = self.train_transforms(spec)
            spec = self.spec_aug(spec)
            if self.use_top4_method:
                spec = self._apply_top4_specaug(spec)

        if self.resize_transforms is not None:
            if self.use_top4_method:
                s_np = spec.detach().cpu().numpy()
                _c, _h, _w = s_np.shape
                out_h, out_w = (
                    int(self.mel_spec_params["mel_image_size"][0]),
                    int(self.mel_spec_params["mel_image_size"][1]),
                )
                resized = cv2.resize(s_np[0], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                spec = torch.from_numpy(resized).to(spec.device, dtype=spec.dtype).unsqueeze(0)
            else:
                spec = self.resize_transforms(spec)

        return spec

    def _resolve_top4_preset(self, config) -> str | None:
        preset = self.mel_spec_params.get("top4_preset")
        if not self.use_top4_method:
            return None
        if preset and str(preset).lower() != "auto":
            return str(preset)
        ck = config.get("model", {}).get("backbone", {}).get("init_checkpoint")
        name = str(ck or "").lower()
        if "v3_first10" in name:
            return "v3_first10"
        if "v2_first10" in name:
            return "v2_first10"
        if "v1_first10" in name:
            return "v1_first10"
        if "v1_full" in name:
            return "v1_full"
        # default top4 family
        return "v1_first10"

    def top4_mel_frontend(self, audio: torch.Tensor) -> torch.Tensor:
        # input: (1, T) waveform tensor
        wav = audio.detach().cpu().numpy().astype("float32").reshape(-1)
        if self.top4_preset in {"v3_first10"}:
            n_fft, hop, n_mels, fmin, fmax = 2048, 128, 224, 40, 16000
        elif self.top4_preset in {"v2_first10"}:
            n_fft, hop, n_mels, fmin, fmax = 1536, 64, 192, 50, 16000
        else:
            n_fft, hop, n_mels, fmin, fmax = 2048, 64, 256, 60, 16000
        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=int(self.mel_spec_params["sample_rate"]),
            n_fft=int(n_fft),
            hop_length=int(hop),
            n_mels=int(n_mels),
            fmin=float(fmin),
            fmax=float(fmax),
            power=2.0,
        )
        mel_db = librosa.power_to_db(mel, ref=1.0)
        spec = torch.from_numpy(mel_db).to(audio.device, dtype=audio.dtype).unsqueeze(0)
        return spec

    def normalize_melspec(self, X):
        if self.norm_method == "none":
            return X
        if self.norm_method == "db_scaled":
            return torch.clamp((X + self.top_db) / self.top_db, 0.0, 1.0)
        elif self.norm_method == "per_sample_minmax":
            lo, hi = X.min(), X.max()
            return (X - lo) / (hi - lo).clamp_min(1e-8)
        elif self.norm_method == "per_sample_absmax":
            return X / X.abs().max().clamp_min(1e-8)
        elif self.norm_method == "per_sample_std":
            return (X - X.mean()) / X.std().clamp_min(1e-6)
        elif self.norm_method == "z_score_ft_plus_minmax":
            return self._normalize_melspec_z_score_ft_plus_minmax(X)
        else:
            raise NotImplementedError

    def _normalize_melspec_z_score_ft_plus_minmax(self, X: torch.Tensor) -> torch.Tensor:
        eps = self.normalize_melspec_eps
        if X.dim() == 3:
            x4 = X.unsqueeze(1)
        elif X.dim() == 4:
            x4 = X
        else:
            raise ValueError(f"z_score_ft_plus_minmax expects 3D or 4D mel, got {tuple(X.shape)}")

        mean = x4.mean((1, 2), keepdim=True)
        std = x4.std((1, 2), keepdim=True)
        xstd = (x4 - mean) / (std + eps)
        if self.normalize_melspec_exportable:
            norm_max = torch.amax(xstd, dim=(1, 2), keepdim=True)
            norm_min = torch.amin(xstd, dim=(1, 2), keepdim=True)
            out = (xstd - norm_min) / (norm_max - norm_min + eps)
        else:
            norm_min, norm_max = (
                xstd.min(-1)[0].min(-1)[0],
                xstd.max(-1)[0].max(-1)[0],
            )
            fix_ind = (norm_max - norm_min) > eps * torch.ones_like(norm_max - norm_min)
            v = torch.zeros_like(xstd)
            if fix_ind.sum():
                v_fix = xstd[fix_ind]
                norm_max_fix = norm_max[fix_ind, None, None]
                norm_min_fix = norm_min[fix_ind, None, None]
                v_fix = torch.max(
                    torch.min(v_fix, norm_max_fix),
                    norm_min_fix,
                )
                v_fix = (v_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
                v[fix_ind] = v_fix
            out = v

        if X.dim() == 3:
            return out.squeeze(1)
        return out
