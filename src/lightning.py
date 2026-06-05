import json
import math
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import LightningModule

from src.awp import AWP
from src.dataset import build_dataloader, build_train_val_dataloaders
from src.metric import macro_auc_skip_missing
from src.models import (
    SEDModelClassic,
    SEDModelCNN,
    SEDModelMultiKHead,
    SEDModelMultiTrans,
    SEDModelPerch,
    SEDModelProtoSED,
    multilabel_bernoulli_kl_mean,
)
from src.optim import get_optimizer, get_scheduler
from src.pretrain_model import PretrainArcFaceModel, PretrainPerchDistillModel
from src.rdrop import rdrop_losses
from src.samplers import get_sampler


def build_model(cfg: dict):
    mtype = cfg["model"]["model_type"]
    if mtype == "sed":
        return SEDModelClassic(cfg)
    if mtype == "proto_sed":
        return SEDModelProtoSED(cfg)
    if mtype in ("cnn", "sed_cnn"):
        return SEDModelCNN(cfg)
    if mtype == "multised":
        return SEDModelMultiKHead(cfg)
    if mtype == "multised_trans":
        return SEDModelMultiTrans(cfg)
    if mtype == "perch_sed":
        return SEDModelPerch(cfg)
    if mtype == "pretrain_arcface":
        return PretrainArcFaceModel(cfg)
    if mtype == "pretrain_perch_distill":
        return PretrainPerchDistillModel(cfg)
    raise NotImplementedError


def other_train_mixup_supervised_loss_from_out(model, out, y, y_bins):
    if isinstance(model, (PretrainArcFaceModel, PretrainPerchDistillModel)):
        return None

    if isinstance(model, SEDModelProtoSED):
        return model.criterion(out["clipwise_logit"], y)

    if isinstance(model, SEDModelCNN):
        return model.criterion(out["clipwise_logit"], y)

    if isinstance(model, SEDModelClassic):
        clipwise_logit = out["clipwise_logit"]
        agg_logit = out["agg_logit"]
        segmentwise_logit = out["segmentwise_logit"]
        if (
            (
                bool(model.config["model"].get("multicontext", False))
                or bool(model.config["model"].get("chunked_multicontext", False))
            )
            and y_bins is not None
            and y_bins.ndim == 3
            and y_bins.size(1) > 0
        ):
            agg_loss = model.criterion(agg_logit, y)
            per_bin_logits = model.segmentwise_bin_max_logits(segmentwise_logit, y_bins.size(1))
            if per_bin_logits is not None:
                k_eff = min(per_bin_logits.size(1), y_bins.size(1))
                if k_eff > 0:
                    per_bin_loss = [model.criterion(per_bin_logits[:, i, :], y_bins[:, i, :]) for i in range(k_eff)]
                    agg_loss = torch.stack(per_bin_loss).mean()
            main_loss = 0.5 * model.criterion(clipwise_logit, y) + 0.5 * agg_loss
        else:
            main_loss = 0.5 * model.criterion(clipwise_logit, y) + 0.5 * model.criterion(agg_logit, y)
            if model.use_symm_kl_div and model.training and not bool(model.config["model"].get("multicontext", False)):
                teacher_p = y.clamp(min=1e-7, max=1.0 - 1e-7)
                student_clip = torch.sigmoid(clipwise_logit).clamp(min=1e-7, max=1.0 - 1e-7)
                student_agg = torch.sigmoid(agg_logit).clamp(min=1e-7, max=1.0 - 1e-7)
                kl_clipwise_ts = multilabel_bernoulli_kl_mean(teacher_p, student_clip)
                kl_agg_ts = multilabel_bernoulli_kl_mean(teacher_p, student_agg)
                kl_ts = (kl_clipwise_ts + kl_agg_ts) / 2.0
                kl_clipwise_st = multilabel_bernoulli_kl_mean(student_clip, teacher_p)
                kl_agg_st = multilabel_bernoulli_kl_mean(student_agg, teacher_p)
                kl_st = (kl_clipwise_st + kl_agg_st) / 2.0
                main_loss = (main_loss + kl_ts + kl_st) / 3.0
        return main_loss

    if isinstance(model, SEDModelMultiKHead):
        per_bin_agg_logit = out["per_bin_agg_logit"]
        clipwise_logit = out["clipwise_logit"]
        agg_logit = out["agg_logit"]
        if y_bins is not None and y_bins.ndim == 3 and y_bins.size(1) > 0:
            k_eff = min(per_bin_agg_logit.size(1), y_bins.size(1))
            if k_eff > 0:
                main_loss = torch.stack(
                    [model.criterion(per_bin_agg_logit[:, i, :], y_bins[:, i, :]) for i in range(k_eff)]
                ).mean()
            else:
                main_loss = model.criterion(agg_logit, y)
        else:
            main_loss = 0.5 * model.criterion(clipwise_logit, y) + 0.5 * model.criterion(agg_logit, y)
        return main_loss

    if isinstance(model, SEDModelMultiTrans):
        per_bin_agg_logit = out["per_bin_agg_logit"]
        clipwise_logit = out["clipwise_logit"]
        agg_logit = out["agg_logit"]
        if y_bins is not None and y_bins.ndim == 3 and y_bins.size(1) > 0:
            k_eff = min(per_bin_agg_logit.size(1), y_bins.size(1))
            if k_eff > 0:
                main_loss = torch.stack(
                    [model.criterion(per_bin_agg_logit[:, i, :], y_bins[:, i, :]) for i in range(k_eff)]
                ).mean()
            else:
                main_loss = model.criterion(agg_logit, y)
        else:
            main_loss = 0.5 * model.criterion(clipwise_logit, y) + 0.5 * model.criterion(agg_logit, y)
        return main_loss

    if isinstance(model, SEDModelPerch):
        clipwise_logit = out["clipwise_logit"]
        agg_logit = out["agg_logit"]
        segmentwise_logit = out["segmentwise_logit"]
        if (
            (
                bool(model.config["model"].get("multicontext", False))
                or bool(model.config["model"].get("chunked_multicontext", False))
            )
            and y_bins is not None
            and y_bins.ndim == 3
            and y_bins.size(1) > 0
        ):
            agg_loss = model.criterion(agg_logit, y)
            per_bin_logits = model.segmentwise_bin_max_logits(segmentwise_logit, y_bins.size(1))
            if per_bin_logits is not None:
                k_eff = min(per_bin_logits.size(1), y_bins.size(1))
                if k_eff > 0:
                    per_bin_loss = [model.criterion(per_bin_logits[:, i, :], y_bins[:, i, :]) for i in range(k_eff)]
                    agg_loss = torch.stack(per_bin_loss).mean()
            return agg_loss
        return 0.5 * model.criterion(clipwise_logit, y) + 0.5 * model.criterion(agg_logit, y)

    return None


class BirdClefLightningModule(LightningModule):
    def __init__(self, cfg: dict, *, total_optimizer_steps, fold_id):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.fold_id = fold_id
        self.total_optimizer_steps = total_optimizer_steps
        self.model = build_model(cfg)
        self.use_awp = cfg["use_awp"]
        self.use_rdrop = cfg["use_rdrop"]
        self.rdrop_alpha = cfg["rdrop_alpha"]
        self.automatic_optimization = not self.use_awp
        self.awp = AWP(self.model, adv_lr=cfg["awp_lr"], adv_eps=cfg["awp_eps"]) if self.use_awp else None
        self.val_clip_preds = []
        self.val_agg_preds = []
        self.val_clip_targets = []
        self.val_agg_targets = []

    def _forward_model(
        self,
        x,
        y=None,
        y_bins=None,
        perch_wave=None,
        *,
        distill_detach_override=None,
        distill_coef_override=None,
        skip_online_augmentations=False,
    ):
        if isinstance(
            self.model,
            (SEDModelClassic, SEDModelProtoSED, SEDModelPerch, SEDModelMultiKHead, SEDModelMultiTrans),
        ):
            return self.model(
                x,
                y,
                y_bins=y_bins,
                perch_wave=perch_wave,
                distill_detach_override=distill_detach_override,
                distill_coef_override=distill_coef_override,
                skip_online_augmentations=skip_online_augmentations,
            )
        if isinstance(self.model, PretrainPerchDistillModel):
            return self.model(
                x,
                y,
                perch_wave=perch_wave,
                skip_online_augmentations=skip_online_augmentations,
            )
        return self.model(x, y)

    def _distill_train_controls(self):
        ratio = float(self.cfg.get("detach_turn_off_ratio", 1.0))
        ratio = max(0.0, min(1.0, ratio))
        total = max(1, int(self.total_optimizer_steps))
        progress = min(1.0, max(0.0, float(self.global_step) / float(total)))
        detach_on = progress < ratio

        coef_max = float(self.cfg.get("distill_perch_coef", 1.0))
        coef_min = float(self.cfg.get("distill_perch_coef_min", coef_max))
        sched = str(self.cfg.get("distill_perch_coef_scheduler", "none")).lower()
        if sched == "none":
            coef = coef_max
        else:
            if sched == "linear":
                f = progress
            elif sched == "cosine":
                f = 0.5 * (1.0 - math.cos(math.pi * progress))
            else:
                raise ValueError(
                    f"Unsupported cfg.distill_perch_coef_scheduler={sched!r}. Use 'none', 'linear', or 'cosine'."
                )
            coef = coef_max + (coef_min - coef_max) * f
        return detach_on, float(coef)

    def _unpack_batch(self, batch):
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Unexpected batch format")
        x, y = batch[0], batch[1]
        y_bins = None
        perch_wave = None
        for extra in batch[2:]:
            if not torch.is_tensor(extra):
                continue
            if extra.dtype == torch.bool:
                continue
            if extra.dim() == 3 and extra.size(1) == 1:
                perch_wave = extra
                continue
            if extra.dim() >= 3:
                y_bins = extra
        return x, y, y_bins, perch_wave

    def _split_segmentwise_to_bins(self, seglog):
        chunk_s = float(self.cfg["dataset"]["chunk_duration_s"])
        bin_s = float(self.cfg["dataset"].get("soundscape_label_bin_s", 5.0))
        k = round(chunk_s / bin_s) if bin_s > 0 else 1
        t = seglog.size(1)
        edges = torch.linspace(0, t, steps=max(2, k + 1), device=seglog.device).round().long()
        per_bin = []
        for i in range(k):
            l = int(edges[i].item())
            r = int(edges[i + 1].item())
            if r <= l:
                continue
            per_bin.append(seglog[:, l:r, :].max(dim=1)[0])
        if not per_bin:
            return None
        return torch.stack(per_bin, dim=1)  # (B, k_eff, C)

    def _validation_agg_probs(self, out):
        if out.get("per_bin_agg_logit") is not None:
            pl = out["per_bin_agg_logit"]
            agg_logit = pl.max(dim=1)[0]
            return torch.sigmoid(agg_logit)

        seglog = out.get("segmentwise_logit")
        if seglog is None:
            return out.get("agg_output", out["clipwise_output"])

        pool = str(self.cfg.get("val_split_pool", "all"))
        chunk_s = float(self.cfg["dataset"]["chunk_duration_s"])
        bin_s = float(self.cfg["dataset"].get("soundscape_label_bin_s", 5.0))
        k = round(chunk_s / bin_s) if bin_s > 0 else 1

        if pool != "soundscape" or k <= 1:
            return out.get("agg_output", out["clipwise_output"])

        t = seglog.size(1)
        if t <= 1:
            return out.get("agg_output", out["clipwise_output"])

        per_bin_max = self._split_segmentwise_to_bins(seglog)
        if per_bin_max is None:
            return out.get("agg_output", out["clipwise_output"])
        agg_logit = per_bin_max.max(dim=1)[0]  # (B, C)
        return torch.sigmoid(agg_logit)

    def forward(self, x, y=None):
        return self.model(x, y)

    def train_batch_xy(self, x, y, y_bins=None):
        if not self.use_rdrop or not self.training:
            return x, y, y_bins
        if isinstance(self.model, SEDModelMultiTrans):
            return self.model.apply_online_augmentations(x, y, y_bins)
        x, y = self.model.apply_online_augmentations(x, y)
        return x, y, y_bins

    def _sample_other_train_mixup(self):
        if not self.training:
            return None
        if not bool(self.cfg.get("other_train_mixup", False)):
            return None
        p = float(self.cfg.get("other_train_mixup_p", 0.0))
        if p <= 0.0 or np.random.rand() >= p:
            return None
        alpha = float(self.cfg.get("other_train_mixup_alpha", 0.4))
        if alpha <= 0:
            return None
        lam = float(np.random.beta(alpha, alpha))
        return lam

    @staticmethod
    def _permute_tensor_or_none(tensor, indices):
        if tensor is None:
            return None
        return tensor[indices]

    def _log_proto_ortho_loss(self, out: dict, batch_size: int, *, suffix: str = "") -> None:
        v = out.get("proto_ortho_loss")
        if v is None:
            return
        tag = "train_proto_ortho_loss" + suffix
        self.log(tag, v, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

    def _log_proto_ortho_loss_pair(self, out1: dict, out2: dict, batch_size: int, *, suffix: str = "") -> None:
        v1 = out1.get("proto_ortho_loss")
        if v1 is None:
            return
        v2 = out2.get("proto_ortho_loss")
        v = 0.5 * (v1 + v2) if v2 is not None else v1
        tag = "train_proto_ortho_loss" + suffix
        self.log(tag, v, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)

    def _mixup_weighted_loss(
        self,
        x,
        y,
        y_bins,
        perch_wave,
        distill_detach,
        distill_coef,
        *,
        skip_online_augmentations=False,
    ):
        lam = self._sample_other_train_mixup()
        if lam is None:
            out = self._forward_model(
                x,
                y,
                y_bins=y_bins,
                perch_wave=perch_wave,
                distill_detach_override=distill_detach,
                distill_coef_override=distill_coef,
                skip_online_augmentations=skip_online_augmentations,
            )
            return out["loss"], out

        indices = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1.0 - lam) * x[indices]
        y_perm = y[indices]
        y_bins_perm = self._permute_tensor_or_none(y_bins, indices)
        perch_wave_perm = self._permute_tensor_or_none(perch_wave, indices)

        out = self._forward_model(
            x_mix,
            y,
            y_bins=y_bins,
            perch_wave=perch_wave,
            distill_detach_override=distill_detach,
            distill_coef_override=distill_coef,
            skip_online_augmentations=skip_online_augmentations,
        )
        lam_t = out["loss"].new_tensor(lam)
        main_perm = other_train_mixup_supervised_loss_from_out(self.model, out, y_perm, y_bins_perm)
        if main_perm is None:
            out_b = self._forward_model(
                x_mix,
                y_perm,
                y_bins=y_bins_perm,
                perch_wave=perch_wave_perm,
                distill_detach_override=distill_detach,
                distill_coef_override=distill_coef,
                skip_online_augmentations=skip_online_augmentations,
            )
            loss = lam_t * out["loss"] + (1.0 - lam_t) * out_b["loss"]
            out_m = dict(out)
            out_m["loss"] = loss
            if out.get("distill_loss") is not None and out_b.get("distill_loss") is not None:
                out_m["distill_loss"] = lam_t * out["distill_loss"] + (1.0 - lam_t) * out_b["distill_loss"]
            return loss, out_m

        d = out.get("distill_loss")
        coef = (
            float(distill_coef) if distill_coef is not None else float(getattr(self.model, "distill_perch_coef", 0.0))
        )
        if d is not None:
            loss = lam_t * out["loss"] + (1.0 - lam_t) * (main_perm + coef * d)
        else:
            loss = lam_t * out["loss"] + (1.0 - lam_t) * main_perm
        out_m = dict(out)
        out_m["loss"] = loss
        return loss, out_m

    def training_step(self, batch, batch_idx):
        x, y, y_bins, perch_wave = self._unpack_batch(batch)
        bs = x.size(0)
        distill_detach, distill_coef = self._distill_train_controls()

        if not self.use_awp:
            if not self.use_rdrop:
                loss, out = self._mixup_weighted_loss(
                    x,
                    y,
                    y_bins,
                    perch_wave,
                    distill_detach,
                    distill_coef,
                )
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
                if out.get("distill_loss") is not None:
                    self.log(
                        "train_distill_loss",
                        out["distill_loss"],
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=bs,
                    )
                    self.log("train_distill_coef", distill_coef, on_step=True, on_epoch=True, batch_size=bs)
                self._log_proto_ortho_loss(out, bs)
                return loss

            xa, ya, y_bins_a = self.train_batch_xy(x, y, y_bins)
            out1 = self._forward_model(
                xa,
                ya,
                y_bins=y_bins_a,
                perch_wave=perch_wave,
                distill_detach_override=distill_detach,
                distill_coef_override=distill_coef,
                skip_online_augmentations=True,
            )
            out2 = self._forward_model(
                xa,
                ya,
                y_bins=y_bins_a,
                perch_wave=perch_wave,
                distill_detach_override=distill_detach,
                distill_coef_override=distill_coef,
                skip_online_augmentations=True,
            )
            loss, sup, kl = rdrop_losses(out1, out2, self.rdrop_alpha)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
            self.log("train_rdrop_sup", sup, on_step=True, on_epoch=True, batch_size=bs)
            self.log("train_rdrop_kl", kl, on_step=True, on_epoch=True, batch_size=bs)
            self._log_proto_ortho_loss_pair(out1, out2, bs)
            return loss

        opt = self.optimizers()
        opt.zero_grad(set_to_none=True)

        if self.use_rdrop:
            xa, ya, y_bins_a = self.train_batch_xy(x, y, y_bins)
            out1 = self._forward_model(
                xa,
                ya,
                y_bins=y_bins_a,
                perch_wave=perch_wave,
                distill_detach_override=distill_detach,
                distill_coef_override=distill_coef,
                skip_online_augmentations=True,
            )
            out2 = self._forward_model(
                xa,
                ya,
                y_bins=y_bins_a,
                perch_wave=perch_wave,
                distill_detach_override=distill_detach,
                distill_coef_override=distill_coef,
                skip_online_augmentations=True,
            )
            loss, sup, kl = rdrop_losses(out1, out2, self.rdrop_alpha)
        else:
            loss, out = self._mixup_weighted_loss(
                x,
                y,
                y_bins,
                perch_wave,
                distill_detach,
                distill_coef,
            )
            sup, kl = loss, loss.new_zeros(())
            distill = out.get("distill_loss")

        self.manual_backward(loss)
        self.awp.attack()

        if self.use_rdrop:
            out1a = self._forward_model(
                xa,
                ya,
                y_bins=y_bins_a,
                perch_wave=perch_wave,
                distill_detach_override=distill_detach,
                distill_coef_override=distill_coef,
                skip_online_augmentations=True,
            )
            out2a = self._forward_model(
                xa,
                ya,
                y_bins=y_bins_a,
                perch_wave=perch_wave,
                distill_detach_override=distill_detach,
                distill_coef_override=distill_coef,
                skip_online_augmentations=True,
            )
            loss_a, sup_a, kl_a = rdrop_losses(out1a, out2a, self.rdrop_alpha)
        else:
            loss_a, out_a = self._mixup_weighted_loss(
                x,
                y,
                y_bins,
                perch_wave,
                distill_detach,
                distill_coef,
            )
            sup_a, kl_a = loss_a, loss_a.new_zeros(())

        self.manual_backward(loss_a)
        self.awp.restore()

        gn = self.cfg["max_norm"]
        if gn > 0:
            self.clip_gradients(opt, gradient_clip_val=gn, gradient_clip_algorithm="norm")
        opt.step()
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

        self.log("train_loss", loss.detach(), on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        self.log("train_loss_awp", loss_a.detach(), on_step=True, on_epoch=True, batch_size=bs)
        if not self.use_rdrop and "distill" in locals() and distill is not None:
            self.log("train_distill_loss", distill.detach(), on_step=True, on_epoch=True, batch_size=bs)
            self.log("train_distill_coef", distill_coef, on_step=True, on_epoch=True, batch_size=bs)
        if self.use_rdrop:
            self.log("train_rdrop_sup", sup.detach(), on_step=True, on_epoch=True, batch_size=bs)
            self.log("train_rdrop_kl", kl.detach(), on_step=True, on_epoch=True, batch_size=bs)
            self.log("train_rdrop_sup_awp", sup_a.detach(), on_step=True, on_epoch=True, batch_size=bs)
            self.log("train_rdrop_kl_awp", kl_a.detach(), on_step=True, on_epoch=True, batch_size=bs)
            self._log_proto_ortho_loss_pair(out1, out2, bs)
            self._log_proto_ortho_loss_pair(out1a, out2a, bs, suffix="_awp")
        else:
            self._log_proto_ortho_loss(out, bs)
            self._log_proto_ortho_loss(out_a, bs, suffix="_awp")

    def on_validation_epoch_start(self):
        self.val_clip_preds.clear()
        self.val_agg_preds.clear()
        self.val_clip_targets.clear()
        self.val_agg_targets.clear()

    def validation_step(self, batch, batch_idx):
        x, y, y_bins, perch_wave = self._unpack_batch(batch)
        out = self._forward_model(x, y, y_bins=y_bins, perch_wave=perch_wave)
        if isinstance(self.model, PretrainPerchDistillModel):
            if out.get("loss") is not None:
                self.log("val_loss", out["loss"], on_step=False, on_epoch=True, batch_size=x.size(0))
            return
        clip_probs = out["clipwise_output"].detach().float().cpu().numpy()
        clip_tgt = y.detach().float().cpu().numpy()
        agg = self._validation_agg_probs(out)
        agg_probs = agg.detach().float().cpu().numpy()
        agg_tgt = clip_tgt

        if y_bins is not None:
            if out.get("per_bin_agg_logit") is not None:
                bin_probs = torch.sigmoid(out["per_bin_agg_logit"]).detach().float().cpu().numpy()
                bin_tgt = y_bins.detach().float().cpu().numpy()
                k_eff = min(bin_probs.shape[1], bin_tgt.shape[1])
                if k_eff > 0:
                    agg_probs = bin_probs[:, :k_eff, :].reshape(-1, bin_probs.shape[-1])
                    agg_tgt = bin_tgt[:, :k_eff, :].reshape(-1, bin_tgt.shape[-1])
            else:
                seglog = out.get("segmentwise_logit")
                if seglog is not None:
                    per_bin_max = self._split_segmentwise_to_bins(seglog)
                    if per_bin_max is not None:
                        bin_probs = torch.sigmoid(per_bin_max).detach().float().cpu().numpy()  # (B, k_eff, C)
                        bin_tgt = y_bins.detach().float().cpu().numpy()  # (B, k, C)
                        k_eff = min(bin_probs.shape[1], bin_tgt.shape[1])
                        if k_eff > 0:
                            agg_probs = bin_probs[:, :k_eff, :].reshape(-1, bin_probs.shape[-1])
                            agg_tgt = bin_tgt[:, :k_eff, :].reshape(-1, bin_tgt.shape[-1])

        self.val_clip_preds.append(clip_probs)
        self.val_agg_preds.append(agg_probs)
        self.val_clip_targets.append(clip_tgt)
        self.val_agg_targets.append(agg_tgt)
        if out.get("distill_loss") is not None:
            self.log("val_distill_loss", out["distill_loss"], on_step=False, on_epoch=True, batch_size=x.size(0))
        if out.get("loss") is not None:
            self.log("val_loss", out["loss"], on_step=False, on_epoch=True, batch_size=x.size(0))
        if out.get("proto_ortho_loss") is not None:
            self.log(
                "val_proto_ortho_loss",
                out["proto_ortho_loss"],
                on_step=False,
                on_epoch=True,
                batch_size=x.size(0),
            )

    def on_validation_epoch_end(self):
        if not self.val_clip_preds:
            return
        y_clip = np.concatenate(self.val_clip_preds, axis=0)
        y_agg = np.concatenate(self.val_agg_preds, axis=0)
        y_true_clip = np.concatenate(self.val_clip_targets, axis=0)
        y_true_agg = np.concatenate(self.val_agg_targets, axis=0)
        auc_clip = macro_auc_skip_missing(y_true_clip, y_clip)
        auc_agg = macro_auc_skip_missing(y_true_agg, y_agg)
        self.log("val_macro_auc", float(auc_clip), prog_bar=True)
        self.log("val_macro_auc_clip", float(auc_clip), prog_bar=False)
        self.log("val_macro_auc_agg", float(auc_agg), prog_bar=True)
        oof_root = Path(self.cfg["oof_dir"])
        oof_root.mkdir(parents=True, exist_ok=True)
        exp = self.cfg["exp_name"]
        np.savez_compressed(
            oof_root / f"{exp}_fold{self.fold_id}_val_probs.npz",
            probs=y_clip,
            clip_probs=y_clip,
            agg_probs=y_agg,
            targets=y_true_clip,
            clip_targets=y_true_clip,
            agg_targets=y_true_agg,
        )
        metrics_path = oof_root / f"{exp}_fold{self.fold_id}_metrics.json"
        prev = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        prev["macro_auc"] = auc_clip
        prev["macro_auc_clip"] = auc_clip
        prev["macro_auc_agg"] = auc_agg
        prev["fold_id"] = self.fold_id
        metrics_path.write_text(json.dumps(prev, indent=2))

    def configure_optimizers(self):
        opt = get_optimizer(self.cfg, self.model)
        sched = get_scheduler(self.cfg, opt, self.total_optimizer_steps)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }


def make_trainer_val_dataloaders(cfg: dict, train_ds, val_ds):
    train_loader, val_loader = build_train_val_dataloaders(cfg, train_ds, val_ds)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * int(cfg["n_epochs"]))
    return train_loader, val_loader, total_steps


def make_trainer_train_only_dataloader(cfg: dict, train_ds):
    sampler = get_sampler(cfg, train_ds)
    train_loader = build_dataloader(cfg, train_ds, sampler, train=True)
    steps_per_epoch = len(train_loader)
    total_steps = max(1, steps_per_epoch * int(cfg["n_epochs"]))
    return train_loader, total_steps
