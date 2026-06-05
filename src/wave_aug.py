import torch


def apply_wave_random_gain_db(
    wav: torch.Tensor,
    min_db: float,
    max_db: float,
    prob: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if prob <= 0.0:
        return wav
    if generator is not None:
        if torch.rand((), generator=generator).item() >= prob:
            return wav
        u = torch.rand((), generator=generator).item()
    else:
        if torch.rand((), device=wav.device, dtype=torch.float32) >= prob:
            return wav
        u = torch.rand((), device=wav.device, dtype=torch.float32).item()
    gain_db = min_db + u * (max_db - min_db)
    return wav * (10 ** (gain_db / 20.0))


def apply_wave_awgn_snr(
    wav: torch.Tensor,
    min_snr_db: float,
    max_snr_db: float,
    prob: float,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if prob <= 0.0:
        return wav
    if generator is not None:
        if torch.rand((), generator=generator).item() >= prob:
            return wav
        u = torch.rand((), generator=generator).item()
    else:
        if torch.rand((), device=wav.device, dtype=torch.float32) >= prob:
            return wav
        u = torch.rand((), device=wav.device, dtype=torch.float32).item()
    sp = (wav * wav).mean()
    if sp <= 1e-10:
        return wav
    lo, hi = min_snr_db, max_snr_db
    if hi < lo:
        lo, hi = hi, lo
    snr_db = lo + u * (hi - lo)
    noise_std = torch.sqrt(sp / (10 ** (snr_db / 10.0)))
    if generator is not None:
        noise = torch.randn(wav.shape, device=wav.device, dtype=wav.dtype, generator=generator)
    else:
        noise = torch.randn_like(wav)
    return wav + noise * noise_std


def apply_wave_aug_like_dataset(
    wav: torch.Tensor,
    wave_aug: dict,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    w = apply_wave_random_gain_db(
        wav,
        float(wave_aug.get("random_gain_min_db", -6.0)),
        float(wave_aug.get("random_gain_max_db", 6.0)),
        float(wave_aug.get("random_gain_prob", 0.0)),
        generator=generator,
    )
    w = apply_wave_awgn_snr(
        w,
        float(wave_aug.get("gaussian_noise_min_snr_db", 10.0)),
        float(wave_aug.get("gaussian_noise_max_snr_db", 30.0)),
        float(wave_aug.get("gaussian_noise_prob", 0.0)),
        generator=generator,
    )
    return w
