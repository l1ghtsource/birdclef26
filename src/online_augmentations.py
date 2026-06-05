import numpy as np
import torch

from src.wave_aug import apply_wave_aug_like_dataset

# TODO: zebra mixup, mixup only same subtype of animals etc


def mixup_partner_indices_species_uniform(targets: torch.Tensor, *, active_eps: float = 0.5) -> torch.Tensor:
    if targets.dim() != 2:
        raise ValueError(f"mixup_partner_indices_species_uniform expects targets (B, C), got {tuple(targets.shape)}")
    b, _c = targets.shape
    device = targets.device
    if b <= 1:
        return torch.zeros(b, dtype=torch.long, device=device)
    active = targets > float(active_eps)
    batch_classes = active.any(dim=0).nonzero(as_tuple=True)[0]
    if batch_classes.numel() == 0:
        return torch.randperm(b, device=device)
    k = int(batch_classes.numel())
    choice_of_class = torch.randint(0, k, (b,), device=device)
    chosen_c = batch_classes[choice_of_class]
    chosen_exp = chosen_c.unsqueeze(1).expand(b, b)
    j_grid = torch.arange(b, device=device).unsqueeze(0).expand(b, b)
    m = active[j_grid, chosen_exp]
    m = m & ~torch.eye(b, dtype=torch.bool, device=device)
    probs = m.to(dtype=torch.float32)
    row_sum = probs.sum(dim=1)
    not_self = ~torch.eye(b, dtype=torch.bool, device=device)
    uni = not_self.to(dtype=torch.float32)
    uni = uni / uni.sum(dim=1, keepdim=True).clamp(min=1e-8)
    probs = torch.where((row_sum > 0).unsqueeze(1), probs, uni)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.multinomial(probs, num_samples=1, replacement=True).squeeze(-1)


def get_second_pair(
    data,
    targets,
    bank_x,
    bank_y,
    ss_bank_share: float,
    *,
    apply_ss_bank_wave_aug: bool = False,
    wave_aug: dict | None = None,
    mixup_balancing: bool = False,
    mixup_balance_active_eps: float = 0.5,
):
    b = data.size(0)
    device = data.device
    if mixup_balancing and targets.dim() == 2:
        perm = mixup_partner_indices_species_uniform(targets, active_eps=mixup_balance_active_eps)
    else:
        perm = torch.randperm(b, device=device)
    from_batch_d = data[perm]
    from_batch_t = targets[perm]
    if bank_x is None or bank_y is None or ss_bank_share <= 0.0:
        return from_batch_d, from_batch_t
    n_bank = bank_x.size(0)
    bx = bank_x.to(device=device, dtype=data.dtype)
    by = bank_y.to(device=device, dtype=targets.dtype)
    use_b = torch.rand(b, device=device) < float(ss_bank_share)
    idx = torch.randint(0, n_bank, (b,), device=device)
    bx_sel = bx[idx]
    if apply_ss_bank_wave_aug and wave_aug is not None and bool(use_b.any().item()):
        bx_aug = bx_sel.clone()
        sel = use_b.nonzero(as_tuple=True)[0]
        for ii in sel.tolist():
            bx_aug[ii] = apply_wave_aug_like_dataset(bx_sel[ii], wave_aug, generator=None)
        bx_sel = bx_aug
    mb = use_b.reshape(b, *([1] * (data.dim() - 1)))
    tb = use_b.unsqueeze(-1).expand_as(targets)
    d2 = torch.where(mb, bx_sel, from_batch_d)
    t2 = torch.where(tb, by[idx], from_batch_t)
    return d2, t2


def mixup(
    data,
    targets,
    alpha,
    use_max_label=False,
    *,
    bank_x=None,
    bank_y=None,
    ss_bank_share: float = 0.0,
    apply_ss_bank_wave_aug: bool = False,
    wave_aug: dict | None = None,
    mixup_balancing: bool = False,
    mixup_balance_active_eps: float = 0.5,
):
    data2, targets2 = get_second_pair(
        data,
        targets,
        bank_x,
        bank_y,
        ss_bank_share,
        apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
        wave_aug=wave_aug,
        mixup_balancing=mixup_balancing,
        mixup_balance_active_eps=mixup_balance_active_eps,
    )
    lam = data.new_tensor(float(np.random.beta(alpha, alpha)))
    data = data * lam + data2 * (1 - lam)
    if not use_max_label:
        targets = targets * lam + targets2 * (1 - lam)
    else:
        targets = torch.maximum(targets, targets2)
    return data, targets


def sumix_freq(
    waves,
    labels,
    max_percent=1.0,
    min_percent=0.3,
    *,
    bank_x=None,
    bank_y=None,
    ss_bank_share: float = 0.0,
    apply_ss_bank_wave_aug: bool = False,
    wave_aug: dict | None = None,
):
    batch_size = labels.size(0)
    device = waves.device
    dtype = waves.dtype
    waves2, labels2 = get_second_pair(
        waves,
        labels,
        bank_x,
        bank_y,
        ss_bank_share,
        apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
        wave_aug=wave_aug,
    )
    span = waves.new_tensor(max_percent - min_percent)
    lo = waves.new_tensor(min_percent)
    coeffs_1 = torch.rand(batch_size, device=device, dtype=dtype).view(-1, 1) * span + lo
    coeffs_2 = torch.rand(batch_size, device=device, dtype=dtype).view(-1, 1) * span + lo
    label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1)).view(-1, 1)
    label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2)).view(-1, 1)
    labels = label_coeffs_1 * labels + label_coeffs_2 * labels2
    if waves.dim() == 4:
        cview = (-1, 1, 1, 1)
    elif waves.dim() == 3:
        cview = (-1, 1, 1)
    else:
        raise ValueError
    waves = coeffs_1.view(cview) * waves + coeffs_2.view(cview) * waves2
    return waves, torch.clip(labels, 0, 1)


def _no_bank_for_stacked_mel(bank_x, ss_bank_share: float, name: str):
    if bank_x is not None and float(ss_bank_share) > 0.0:
        raise NotImplementedError(
            f"{name}: soundscape bank mixing is not supported for stacked mels (B,K,C,H,W); "
            "set online_aug.use_ss_bank=False or ss_bank_share=0."
        )


def mixup_stacked_mel(
    data: torch.Tensor,
    targets: torch.Tensor,
    y_bins: torch.Tensor | None,
    alpha: float,
    use_max_label: bool = False,
    *,
    bank_x=None,
    bank_y=None,
    ss_bank_share: float = 0.0,
    apply_ss_bank_wave_aug: bool = False,
    wave_aug: dict | None = None,
    mixup_balancing: bool = False,
    mixup_balance_active_eps: float = 0.5,
):
    _ = bank_y, apply_ss_bank_wave_aug, wave_aug
    if data.dim() != 5:
        raise ValueError(f"mixup_stacked_mel expects (B,K,C,H,W), got {tuple(data.shape)}")
    _no_bank_for_stacked_mel(bank_x, ss_bank_share, "mixup_stacked_mel")
    b = data.size(0)
    device = data.device
    if mixup_balancing and targets.dim() == 2:
        perm = mixup_partner_indices_species_uniform(targets, active_eps=mixup_balance_active_eps)
    else:
        perm = torch.randperm(b, device=device)
    data2 = data[perm]
    targets2 = targets[perm]
    lam = data.new_tensor(float(np.random.beta(alpha, alpha)))
    data = data * lam + data2 * (1 - lam)
    if not use_max_label:
        targets = targets * lam + targets2 * (1 - lam)
    else:
        targets = torch.maximum(targets, targets2)
    if y_bins is not None:
        y2 = y_bins[perm]
        if not use_max_label:
            y_bins = y_bins * lam + y2 * (1 - lam)
        else:
            y_bins = torch.maximum(y_bins, y2)
    return data, targets, y_bins


def sumix_freq_stacked_mel(
    waves: torch.Tensor,
    labels: torch.Tensor,
    y_bins: torch.Tensor | None = None,
    max_percent: float = 1.0,
    min_percent: float = 0.3,
    *,
    bank_x=None,
    bank_y=None,
    ss_bank_share: float = 0.0,
    apply_ss_bank_wave_aug: bool = False,
    wave_aug: dict | None = None,
):
    _ = bank_y, apply_ss_bank_wave_aug, wave_aug
    if waves.dim() != 5:
        raise ValueError(f"sumix_freq_stacked_mel expects (B,K,C,H,W), got {tuple(waves.shape)}")
    _no_bank_for_stacked_mel(bank_x, ss_bank_share, "sumix_freq_stacked_mel")
    batch_size = labels.size(0)
    device = waves.device
    dtype = waves.dtype
    perm = torch.randperm(batch_size, device=device)
    waves2 = waves[perm]
    labels2 = labels[perm]
    span = waves.new_tensor(max_percent - min_percent)
    lo = waves.new_tensor(min_percent)
    coeffs_1 = torch.rand(batch_size, device=device, dtype=dtype).view(-1, 1) * span + lo
    coeffs_2 = torch.rand(batch_size, device=device, dtype=dtype).view(-1, 1) * span + lo
    label_coeffs_1 = torch.where(coeffs_1 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_1)).view(-1, 1)
    label_coeffs_2 = torch.where(coeffs_2 >= 0.5, 1, 1 - 2 * (0.5 - coeffs_2)).view(-1, 1)
    labels = label_coeffs_1 * labels + label_coeffs_2 * labels2
    cview = (-1, 1, 1, 1, 1)
    waves = coeffs_1.view(cview) * waves + coeffs_2.view(cview) * waves2
    labels = torch.clip(labels, 0, 1)
    if y_bins is not None:
        yb2 = y_bins[perm]
        lc1 = label_coeffs_1.view(-1, 1, 1)
        lc2 = label_coeffs_2.view(-1, 1, 1)
        y_bins = torch.clip(lc1 * y_bins + lc2 * yb2, 0, 1)
    return waves, labels, y_bins


def horizontal_cutmix_mel(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    use_max_label: bool = False,
    *,
    bank_x=None,
    bank_y=None,
    ss_bank_share: float = 0.0,
    mixup_balancing: bool = False,
    mixup_balance_active_eps: float = 0.5,
):
    if x.dim() != 4:
        raise ValueError(f"horizontal_cutmix_mel expects (B,C,H,W), got {tuple(x.shape)}")
    w = int(x.shape[-1])
    if w < 2:
        return x, y
    device = x.device
    b = int(x.size(0))
    x2, y2 = get_second_pair(
        x,
        y,
        bank_x,
        bank_y,
        ss_bank_share,
        apply_ss_bank_wave_aug=False,
        wave_aug=None,
        mixup_balancing=mixup_balancing,
        mixup_balance_active_eps=mixup_balance_active_eps,
    )
    if alpha > 0.0:
        dist = torch.distributions.Beta(
            x.new_tensor(float(alpha)),
            x.new_tensor(float(alpha)),
        )
        lam_frac = dist.sample((b,))
    else:
        lam_frac = torch.rand(b, device=device, dtype=x.dtype)
    cuts = (lam_frac * float(w)).long().clamp(min=1, max=w - 1)
    t = torch.arange(w, device=device, dtype=torch.long).view(1, 1, 1, w)
    mask = (t < cuts.view(b, 1, 1, 1)).to(dtype=x.dtype)
    x_out = x * mask + x2 * (1.0 - mask)
    if not use_max_label:
        lam = cuts.to(dtype=y.dtype) / float(w)
        y = lam.view(b, 1) * y + (1.0 - lam.view(b, 1)) * y2
    else:
        y = torch.maximum(y, y2)
    return x_out, y


def horizontal_cutmix_stacked_mel(
    x: torch.Tensor,
    y: torch.Tensor,
    y_bins: torch.Tensor | None,
    alpha: float,
    use_max_label: bool = False,
    *,
    bank_x=None,
    bank_y=None,
    ss_bank_share: float = 0.0,
    mixup_balancing: bool = False,
    mixup_balance_active_eps: float = 0.5,
):
    _ = bank_y
    _no_bank_for_stacked_mel(bank_x, ss_bank_share, "horizontal_cutmix_stacked_mel")
    if x.dim() != 5:
        raise ValueError(f"horizontal_cutmix_stacked_mel expects (B,K,C,H,W), got {tuple(x.shape)}")
    w = int(x.shape[-1])
    if w < 2:
        return x, y, y_bins
    device = x.device
    b = int(x.size(0))
    if mixup_balancing and y.dim() == 2:
        perm = mixup_partner_indices_species_uniform(y, active_eps=mixup_balance_active_eps)
    else:
        perm = torch.randperm(b, device=device)
    x2 = x[perm]
    y2 = y[perm]
    if alpha > 0.0:
        dist = torch.distributions.Beta(
            x.new_tensor(float(alpha)),
            x.new_tensor(float(alpha)),
        )
        lam_frac = dist.sample((b,))
    else:
        lam_frac = torch.rand(b, device=device, dtype=x.dtype)
    cuts = (lam_frac * float(w)).long().clamp(min=1, max=w - 1)
    t = torch.arange(w, device=device, dtype=torch.long).view(1, 1, 1, 1, w)
    mask = (t < cuts.view(b, 1, 1, 1, 1)).to(dtype=x.dtype)
    x_out = x * mask + x2 * (1.0 - mask)
    if not use_max_label:
        lam = cuts.to(dtype=y.dtype) / float(w)
        lv = lam.view(b, 1)
        y = lv * y + (1.0 - lv) * y2
        if y_bins is not None:
            yb2 = y_bins[perm]
            lv3 = lam.view(b, 1, 1)
            y_bins = lv3 * y_bins + (1.0 - lv3) * yb2
    else:
        y = torch.maximum(y, y2)
        if y_bins is not None:
            yb2 = y_bins[perm]
            y_bins = torch.maximum(y_bins, yb2)
    return x_out, y, y_bins
