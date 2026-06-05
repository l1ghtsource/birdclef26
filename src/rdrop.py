import torch
import torch.nn.functional as F


def symmetric_kl_probs(p1, p2, eps=1e-7):
    p1 = p1.clamp(eps, 1.0 - eps)
    p2 = p2.clamp(eps, 1.0 - eps)
    b = p1.size(0)
    p = torch.stack([1.0 - p1, p1], dim=-1).reshape(-1, 2)
    q = torch.stack([1.0 - p2, p2], dim=-1).reshape(-1, 2)
    kl_12 = F.kl_div(q.log(), p, reduction="sum", log_target=False) / b
    kl_21 = F.kl_div(p.log(), q, reduction="sum", log_target=False) / b
    return 0.5 * (kl_12 + kl_21)


def rdrop_losses(out1, out2, alpha):
    sup = 0.5 * (out1["loss"] + out2["loss"])
    if alpha <= 0:
        kl = sup.new_zeros(())
        return sup, sup, kl
    kl = symmetric_kl_probs(out1["clipwise_output"], out2["clipwise_output"])
    return sup + alpha * kl, sup, kl
