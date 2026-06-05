import torch


# https://arxiv.org/pdf/2004.05884
class AWP:
    def __init__(self, model, adv_lr=1e-3, adv_eps=1e-2):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and param.dtype.is_floating_point:
                self.backup[name] = param.data.clone()
                grad_norm = torch.norm(param.grad)
                if grad_norm != 0 and not torch.isnan(grad_norm):
                    r_at = self.adv_lr * param.grad / (grad_norm + 1e-12)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup[name] - self.adv_eps),
                        self.backup[name] + self.adv_eps,
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
