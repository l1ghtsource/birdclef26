import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# TODO: try supcon loss (ss vs tr audios)

# use data domains as labels (in our case, train data: 0, unlabeled_soundscapes:
# 1), and bottleneck_layer's output as features to do SupCon learning. The total loss is traditional task loss + weight*SupCon loss
# weight	Public	Private
# 1e-2	0.56	0.57
# 1e-3	0.66	0.63
# 1e-4	0.68	0.66
# 1e-5	0.65	0.64
# temperature	Public	Private
# 0.07	0.62	0.62
# 0.09	0.65	0.65
# 0.10	0.68	0.66
# 0.11	0.65	0.66


class FocalLossPlusBCE(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean", bce_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focal_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focal_loss


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(preds, targets)
        probas = torch.sigmoid(preds)
        loss = (
            targets * self.alpha * (1.0 - probas) ** self.gamma * bce_loss
            + (1.0 - targets) * probas**self.gamma * bce_loss
        )
        loss = loss.mean()
        return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.to(dtype=logits.dtype)
        probs = torch.sigmoid(logits)
        xs_pos = probs
        xs_neg = 1.0 - probs

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        pos_loss = targets * torch.log(xs_pos.clamp(min=self.eps))
        neg_loss = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = pos_loss + neg_loss

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = xs_pos * targets + xs_neg * (1.0 - targets)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w

        return (-loss).mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        if targets.dtype in (torch.int32, torch.int64) and targets.dim() == 1:
            return F.cross_entropy(logits, targets)

        soft_targets = targets.to(dtype=logits.dtype).clamp(min=0.0)

        log_probs = F.log_softmax(logits, dim=1)
        return (-(soft_targets * log_probs).sum(dim=1)).mean()


class NormalizedSoftTargetCrossEntropy(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        if targets.dtype in (torch.int32, torch.int64) and targets.dim() == 1:
            return F.cross_entropy(logits, targets)

        t = targets.to(dtype=logits.dtype).clamp(min=0.0)
        row_sum = t.sum(dim=1, keepdim=True).clamp(min=self.eps)
        soft_targets = t / row_sum
        log_probs = F.log_softmax(logits, dim=1)
        return (-(soft_targets * log_probs).sum(dim=1)).mean()


class SoftAUCLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds, labels, sample_weights=None):
        pos_preds = preds[labels > 0.5]
        neg_preds = preds[labels < 0.5]
        pos_labels = labels[labels > 0.5]
        neg_labels = labels[labels < 0.5]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        pos_weights = torch.ones_like(pos_preds) * self.pos_weight * (pos_labels - 0.5)
        neg_weights = torch.ones_like(neg_preds) * self.neg_weight * (0.5 - neg_labels)
        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights] * labels.shape[1], dim=1)
            pos_weights = pos_weights * sample_weights
            neg_weights = neg_weights * sample_weights

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # [N_pos, N_neg]
        loss_matrix = F.softplus(-diff * self.margin)  # torch.log(1 + torch.exp(-diff * self.margin))  # [N_pos, N_neg]
        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)

        return weighted_loss.mean()


def build_loss(cfg):
    loss_name = cfg["loss_name"]
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "focal_loss_plus_bce":
        return FocalLossPlusBCE(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
    elif loss_name == "bce_focal_loss":
        return BCEFocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
    elif loss_name == "asymmetric_loss":
        return AsymmetricLoss(
            gamma_neg=cfg["asymmetric_gamma_neg"], gamma_pos=cfg["asymmetric_gamma_pos"], clip=cfg["asymmetric_clip"]
        )
    elif loss_name == "soft_target_cross_entropy":
        return SoftTargetCrossEntropy()
    elif loss_name == "normalized_ce":
        return NormalizedSoftTargetCrossEntropy()
    elif loss_name == "soft_auc_loss":
        return SoftAUCLoss()
    elif loss_name == "arcface":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    else:
        raise NotImplementedError
