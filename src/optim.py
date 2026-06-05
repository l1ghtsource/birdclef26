import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)


def iter_params(model_or_params):
    if hasattr(model_or_params, "parameters"):
        return model_or_params.parameters()
    return model_or_params


def zeropower_via_newtonschulz5(G, steps: int):
    # https://github.com/KellerJordan/Muon/blob/master/muon.py
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (
                    dist.get_world_size() - len(params) % dist.get_world_size()
                )
                for base_i in range(len(params))[:: dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(
                        params_pad[base_i : base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()]
                    )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad, state["exp_avg"], state["exp_avg_sq"], state["step"], group["betas"], group["eps"]
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


def muon_auxadam_param_groups(params, lr, weight_decay, momentum=0.95, betas=(0.9, 0.95), eps=1e-10):
    params = list(params)
    muon_params = [p for p in params if p.ndim >= 2]
    adam_params = [p for p in params if p.ndim < 2]
    param_groups = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "use_muon": True,
            }
        )
    if adam_params:
        param_groups.append(
            {
                "params": adam_params,
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "use_muon": False,
            }
        )
    if not param_groups:
        raise ValueError("MuonWithAuxAdam: empty parameter list")
    return param_groups


def parse_head_to_bb_lr_ratio(raw) -> tuple[float, float] | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"none", "null"}:
        return None
    if ":" not in s:
        raise ValueError(f"Invalid head_to_bb_lr_ratio={raw!r}. Expected format like '5:1' or '1:1'.")
    h, b = s.split(":", 1)
    h = float(h.strip())
    b = float(b.strip())
    if h <= 0.0 or b <= 0.0:
        raise ValueError(f"Invalid head_to_bb_lr_ratio={raw!r}. Both parts must be > 0.")
    return h, b


def split_backbone_and_head_params(model_or_params):
    if not hasattr(model_or_params, "encoder"):
        return None
    all_params = [p for p in iter_params(model_or_params) if p.requires_grad]
    bb_ids = {id(p) for p in model_or_params.encoder.parameters() if p.requires_grad}
    bb_params = [p for p in all_params if id(p) in bb_ids]
    head_params = [p for p in all_params if id(p) not in bb_ids]
    if not bb_params or not head_params:
        return None
    return bb_params, head_params


def get_optimizer(cfg, model_or_params):
    params = list(iter_params(model_or_params))
    optim_type = cfg["optim_type"]
    lr = cfg["lr"]
    weight_decay = cfg["weight_decay"]
    ratio = parse_head_to_bb_lr_ratio(cfg.get("head_to_bb_lr_ratio"))
    split = split_backbone_and_head_params(model_or_params)
    use_ratio = ratio is not None and split is not None
    if use_ratio:
        head_ratio, bb_ratio = ratio
        bb_params, head_params = split
        head_lr = float(lr)
        bb_lr = float(lr) * (bb_ratio / head_ratio)
    if optim_type == "adamw":
        if use_ratio:
            return torch.optim.AdamW(
                [
                    {"params": bb_params, "lr": bb_lr, "weight_decay": weight_decay},
                    {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
                ]
            )
        arc_margin = getattr(model_or_params, "arc_margin", None)
        if arc_margin is not None and bool(cfg.get("arcface_no_weight_decay_on_margin", True)):
            margin_ids = {id(p) for p in arc_margin.parameters()}
            decay_params = [p for p in params if id(p) not in margin_ids]
            margin_params = [p for p in params if id(p) in margin_ids]
            if margin_params:
                return torch.optim.AdamW(
                    [
                        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
                        {"params": margin_params, "lr": lr, "weight_decay": 0.0},
                    ],
                )
        distill_head = getattr(model_or_params, "distill_head", None)
        if distill_head is not None and bool(cfg.get("perch_distill_no_weight_decay_on_head", True)):
            head_ids = {id(p) for p in distill_head.parameters()}
            decay_params = [p for p in params if id(p) not in head_ids]
            head_params = [p for p in params if id(p) in head_ids]
            if head_params:
                return torch.optim.AdamW(
                    [
                        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
                        {"params": head_params, "lr": lr, "weight_decay": 0.0},
                    ],
                )
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if optim_type == "muon":
        if use_ratio:
            groups = []
            groups.extend(muon_auxadam_param_groups(bb_params, lr=bb_lr, weight_decay=weight_decay))
            groups.extend(muon_auxadam_param_groups(head_params, lr=head_lr, weight_decay=weight_decay))
        else:
            groups = muon_auxadam_param_groups(params, lr=lr, weight_decay=weight_decay)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return MuonWithAuxAdam(groups)
        return SingleDeviceMuonWithAuxAdam(groups)
    if optim_type == "radam":
        if use_ratio:
            return torch.optim.RAdam(
                [
                    {"params": bb_params, "lr": bb_lr, "weight_decay": weight_decay},
                    {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
                ]
            )
        return torch.optim.RAdam(params, lr=lr, weight_decay=weight_decay)
    raise NotImplementedError


def get_scheduler(cfg, optimizer, num_training_steps):
    num_warmup_steps = int(cfg["num_warmup_steps_ratio"] * num_training_steps)
    name = cfg["scheduler"]
    if name == "cosine_anneal_warm_rest_each_5ep":
        n_epochs = max(1, int(cfg["n_epochs"]))
        steps_per_epoch = max(1, int(num_training_steps // n_epochs))
        t0_steps = max(1, steps_per_epoch * 5)
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=t0_steps,
            T_mult=1,
        )
    if name == "cosine_annealing_lr":
        n_epochs = max(1, int(cfg["n_epochs"]))
        steps_per_epoch = max(1, int(num_training_steps // n_epochs))
        t_max_epochs = int(cfg.get("scheduler_t_max_epochs", n_epochs))
        t_max_steps = max(1, t_max_epochs * steps_per_epoch)
        eta_min = float(cfg.get("min_lr", 0.0))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max_steps,
            eta_min=eta_min,
        )
    if name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    if name == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    raise NotImplementedError
