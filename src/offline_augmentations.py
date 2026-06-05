import random

import torch
import torchaudio
import torchvision.transforms.functional as F


class SpecAugment(torch.nn.Module):
    def __init__(self, freq_mask_param=15, time_mask_param=35, num_freq_masks=2, num_time_masks=2, p=1.0):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.p = p

    def forward(self, spec):
        if self.p < 1.0 and random.random() >= self.p:
            return spec
        for _ in range(self.num_freq_masks):
            spec = self.freq_mask(spec)
        for _ in range(self.num_time_masks):
            spec = self.time_mask(spec)
        return spec


class LocalGlobalStretch(torch.nn.Module):
    def __init__(
        self,
        global_stretch_prob=0.5,
        local_stretch_prob=0.5,
        max_global_stretch=0.2,
        max_local_stretch=0.3,
        max_local_regions=3,
    ):
        super().__init__()
        self.global_stretch_prob = global_stretch_prob
        self.local_stretch_prob = local_stretch_prob
        self.max_global_stretch = max_global_stretch
        self.max_local_stretch = max_local_stretch
        self.max_local_regions = max_local_regions

    def _global_stretch(self, spec):
        _, h, w = spec.shape
        stretch_dim = random.randint(0, 2)  # 0: freq, 1: time, 2: both
        new_h = h
        new_w = w
        if stretch_dim in [0, 2]:
            stretch_factor = 1.0 + random.uniform(-self.max_global_stretch, self.max_global_stretch)
            new_h = max(int(h * stretch_factor), 1)

        if stretch_dim in [1, 2]:
            stretch_factor = 1.0 + random.uniform(-self.max_global_stretch, self.max_global_stretch)
            new_w = max(int(w * stretch_factor), 1)

        stretched = F.resize(spec, [new_h, new_w], antialias=True)
        return F.resize(stretched, [h, w], antialias=True)

    def _local_stretch(self, spec):
        _, h, w = spec.shape
        spec_modified = spec.clone()

        num_regions = random.randint(1, self.max_local_regions)

        for _ in range(num_regions):
            is_freq_stretch = random.random() < 0.5
            if is_freq_stretch:
                region_h = random.randint(max(1, int(h * 0.1)), max(2, int(h * 0.5)))
                start_h = random.randint(0, h - region_h)
                region = spec[:, start_h : start_h + region_h, :]
                stretch_factor = 1.0 + random.uniform(-self.max_local_stretch, self.max_local_stretch)
                new_h = max(int(region_h * stretch_factor), 1)
                stretched = F.resize(region, [new_h, w], antialias=True)
                stretched = F.resize(stretched, [region_h, w], antialias=True)
                spec_modified[:, start_h : start_h + region_h, :] = stretched
            else:
                region_w = random.randint(max(1, int(w * 0.1)), max(2, int(w * 0.5)))
                start_w = random.randint(0, w - region_w)
                region = spec[:, :, start_w : start_w + region_w]
                stretch_factor = 1.0 + random.uniform(-self.max_local_stretch, self.max_local_stretch)
                new_w = max(int(region_w * stretch_factor), 1)
                stretched = F.resize(region, [h, new_w], antialias=True)
                stretched = F.resize(stretched, [h, region_w], antialias=True)
                spec_modified[:, :, start_w : start_w + region_w] = stretched

        return spec_modified

    def forward(self, spec):
        if random.random() < self.global_stretch_prob:
            spec = self._global_stretch(spec)
        if random.random() < self.local_stretch_prob:
            spec = self._local_stretch(spec)
        return spec


class AddGaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.01, p: float = 1.0):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, spec):
        if self.p < 1.0 and random.random() >= self.p:
            return spec
        noise = torch.randn_like(spec) * self.std + self.mean
        return spec + noise


class TimeShift(torch.nn.Module):
    def __init__(self, max_shift_pct=0.1, p: float = 1.0):
        super().__init__()
        self.max_shift_pct = max_shift_pct
        self.p = p

    def forward(self, spec):
        if self.p < 1.0 and random.random() >= self.p:
            return spec
        _, _, width = spec.shape
        max_shift = int(width * self.max_shift_pct)
        if max_shift < 1:
            return spec

        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return spec

        return torch.roll(spec, shifts=shift, dims=2)


class FrequencyShift(torch.nn.Module):
    def __init__(self, max_shift_pct=0.05, p: float = 1.0):
        super().__init__()
        self.max_shift_pct = max_shift_pct
        self.p = p

    def forward(self, spec):
        if self.p < 1.0 and random.random() >= self.p:
            return spec
        _, height, _ = spec.shape
        max_shift = int(height * self.max_shift_pct)
        if max_shift < 1:
            return spec

        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return spec

        return torch.roll(spec, shifts=shift, dims=1)


def apply_random_gain_db(tensor, min_db=-6, max_db=6, p: float = 1.0):
    if p < 1.0 and random.random() >= p:
        return tensor
    gain_db = (max_db - min_db) * torch.rand(1).item() + min_db
    gain_linear = 10 ** (gain_db / 20)
    return tensor * gain_linear


# https://github.com/frednam93/FilterAugSED/tree/main
def filt_aug(
    features,
    db_range=None,
    n_band=None,
    min_bw=6,
    filter_type="linear",
    p: float = 1.0,
):
    if p < 1.0 and random.random() >= p:
        return features
    if db_range is None:
        db_range = [-6, 6]
    if n_band is None:
        n_band = [3, 6]
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6
    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()  # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = (
            torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1, (n_freq_band - 1,)))[0]
            + torch.arange(1, n_freq_band) * min_bw
        )
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))
        if filter_type == "step":
            band_factors = (
                torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            )
            band_factors = 10 ** (band_factors / 10)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i] : band_bndry_freqs[i + 1], :] = (
                    band_factors[:, i].unsqueeze(-1).unsqueeze(-1)
                )
        elif filter_type == "linear":
            band_factors = (
                torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            )
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i] : band_bndry_freqs[i + 1], :] = torch.linspace(
                        band_factors[j, i], band_factors[j, i + 1], band_bndry_freqs[i + 1] - band_bndry_freqs[i]
                    ).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 10)
        return features * freq_filt
    else:
        return features
