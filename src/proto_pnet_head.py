import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoPNetHeadTorch(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        num_prototypes: int,
        *,
        non_negative_kernel: bool = True,
        ortho_loss_weight: float = 1.0,
        kernel_init_value: float = 2.0,
        bias_init_value: float = -2.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.num_prototypes = int(num_prototypes)
        self.non_negative_kernel = bool(non_negative_kernel)
        self.ortho_loss_weight = float(ortho_loss_weight)
        self.eps = float(eps)

        self.prototypes = nn.Parameter(torch.empty(self.num_classes, self.embedding_dim, self.num_prototypes))
        nn.init.trunc_normal_(self.prototypes, mean=1.0, std=0.5, a=1e-5, b=2.0)
        self.protop_kernel = nn.Parameter(torch.full((self.num_classes, self.num_prototypes), float(kernel_init_value)))
        self.bias = nn.Parameter(torch.full((self.num_classes,), float(bias_init_value)))

    def _ortho_loss(self, unit_kernel: torch.Tensor) -> torch.Tensor:
        # unit_kernel: (C, D, P) L2-normalized along D
        c, _d, p = unit_kernel.shape
        u = unit_kernel.transpose(1, 2).contiguous()  # (C, P, D)
        proto_sim = torch.bmm(u, unit_kernel)  # (C, P, P)
        eye = torch.eye(p, device=proto_sim.device, dtype=proto_sim.dtype).unsqueeze(0)
        proto_sim = proto_sim - eye
        return self.ortho_loss_weight * (proto_sim**2).sum() / (p * p * c)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unit_kernel = F.normalize(self.prototypes, dim=1, eps=self.eps)
        ortho = self._ortho_loss(unit_kernel)
        unit_inputs = F.normalize(inputs, dim=-1, eps=self.eps)
        sims = torch.einsum("bhwd,cdp->bhwcp", unit_inputs, unit_kernel)
        sims = sims.amax(dim=(1, 2))
        pk = torch.clamp(self.protop_kernel, min=0.0) if self.non_negative_kernel else self.protop_kernel
        logits = (sims * pk).sum(dim=-1) + self.bias
        return logits, ortho
