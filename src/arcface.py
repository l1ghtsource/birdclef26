import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(self, embedding_size: int, num_classes: int, *, s: float = 64.0, m: float = 0.50):
        super().__init__()
        self.s = float(s)
        self.m = float(m)
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self._cos_m = math.cos(self.m)
        self._sin_m = math.sin(self.m)
        self._th = math.cos(math.pi - self.m)
        self._mm = math.sin(math.pi - self.m) * self.m

    def _stable_logits(self, emb: torch.Tensor, targets: torch.Tensor | None) -> torch.Tensor:
        dtype = emb.dtype
        emb_f = emb.float()
        w_f = self.weight.float()
        emb_n = F.normalize(emb_f, p=2, dim=1, eps=self.eps)
        w_n = F.normalize(w_f, p=2, dim=1, eps=self.eps)
        cosine = torch.mm(emb_n, w_n.t())

        if targets is None:
            logits = cosine * self.s
        else:
            targets = targets.long().view(-1)
            cosine_c = cosine.clamp(-1.0 + self.eps, 1.0 - self.eps)
            batch_size = emb.size(0)
            idx = torch.arange(batch_size, device=emb.device)

            cos_gt = cosine_c[idx, targets]
            sin_gt = torch.sqrt((1.0 - cos_gt * cos_gt).clamp(min=self.eps, max=1.0 - self.eps))
            phi_gt = cos_gt * self._cos_m - sin_gt * self._sin_m
            phi_gt = torch.where(cos_gt > self._th, phi_gt, cos_gt - self._mm)
            phi_gt = phi_gt.clamp(-1.0 + self.eps, 1.0 - self.eps)

            logits = cosine_c * self.s
            logits[idx, targets] = phi_gt * self.s

        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        if dtype != logits.dtype:
            logits = logits.to(dtype)
        return logits

    def forward(self, emb: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        return self._stable_logits(emb, targets)
