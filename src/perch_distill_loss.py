import torch
import torch.nn.functional as F


def perch_embedding_distill_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    *,
    alpha: float,
    coef: float = 1.0,
    do_norm: bool = True,
) -> torch.Tensor:
    a = float(alpha)
    a = max(0.0, min(1.0, a))
    s_in = torch.nan_to_num(student_emb.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    t_in = torch.nan_to_num(teacher_emb.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    if do_norm:
        s = F.normalize(s_in, p=2, dim=1, eps=1e-6)
        t = F.normalize(t_in, p=2, dim=1, eps=1e-6)
    else:
        s = s_in
        t = t_in
    mse = F.mse_loss(s, t)
    cos = F.cosine_similarity(s, t, dim=1, eps=1e-6)
    cos_loss = (1.0 - cos.clamp(-1.0, 1.0)).mean()
    combined = a * mse + (1.0 - a) * cos_loss
    out = combined * float(coef)
    return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=0.0)
