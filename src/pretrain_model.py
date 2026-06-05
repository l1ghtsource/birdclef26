from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arcface import ArcMarginProduct
from src.blocks import init_layer
from src.losses import build_loss
from src.perch import HybridPerchExtractor
from src.perch_distill_loss import perch_embedding_distill_loss


def get_timm_model(backbone_config, *, in_chans: int = 3):
    return timm.create_model(
        backbone_config["backbone_name"],
        pretrained=backbone_config["pretrained"],
        pretrained_cfg_overlay={"file": f"{backbone_config['backbone_name']}/model.safetensors"},
        drop_rate=backbone_config["drop_rate"],
        drop_path_rate=backbone_config["drop_path_rate"],
        num_classes=0,
        global_pool="avg",
        in_chans=int(in_chans),
    )


def infer_timm_encoder_embed_dim(encoder: nn.Module, config: dict) -> int:
    msp = config["mel_spec_params"]
    h, w = int(msp["mel_image_size"][0]), int(msp["mel_image_size"][1])
    in_chans = int(config.get("model", {}).get("num_channels", 3))
    encoder.eval()
    with torch.inference_mode():
        x = torch.zeros(1, in_chans, h, w)
        y = encoder(x)
    if y.dim() == 4:
        y = y.mean(dim=(2, 3))
    if y.dim() != 2:
        raise ValueError(f"Expected encoder output (N, D) or (N, C, H, W), got shape {tuple(y.shape)}")
    return int(y.shape[1])


class PretrainArcFaceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_chans = int(self.config.get("model", {}).get("num_channels", 3))
        self.encoder = get_timm_model(self.config["model"]["backbone"], in_chans=in_chans)
        in_features = infer_timm_encoder_embed_dim(self.encoder, self.config)

        af = self.config.get("arcface") or {}
        self.arc_margin = ArcMarginProduct(
            embedding_size=in_features,
            num_classes=self.config["num_classes"],
            s=float(af.get("s", 64.0)),
            m=float(af.get("m", 0.50)),
        )
        self.criterion = build_loss(self.config)

    def apply_online_augmentations(self, x, y):
        return x, y

    def _targets_from_multihot(self, y):
        t = torch.argmax(y, dim=1).long()
        nc = int(self.config["num_classes"])
        if nc > 0:
            t = t.clamp(0, nc - 1)
        return t

    def forward(self, x, y=None, *, skip_online_augmentations=False):
        feats = self.encoder(x)

        if y is None:
            logits = self.arc_margin(feats, targets=None)
            loss = None
        else:
            targets = self._targets_from_multihot(y)
            logits = self.arc_margin(feats, targets=targets)
            loss = self.criterion(logits, targets)

        probs = F.softmax(logits, dim=1)
        return {
            "framewise_output": None,
            "framewise_logit": None,
            "segmentwise_output": None,
            "clipwise_output": probs,
            "clipwise_logit": logits,
            "logit": logits,
            "loss": loss,
        }


class PretrainPerchDistillModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        in_chans = int(self.config.get("model", {}).get("num_channels", 3))
        self.encoder = get_timm_model(self.config["model"]["backbone"], in_chans=in_chans)
        in_features = infer_timm_encoder_embed_dim(self.encoder, self.config)
        self.distill_head = nn.Linear(in_features, 1536, bias=True)
        br = Path(__file__).resolve().parents[1] / "models" / "pytorch_perchv2"
        onnx_p = Path(self.config.get("distill_perch_onnx_path", br / "perch_v2_no_dft.onnx"))
        fe_p = Path(self.config.get("distill_perch_pytorch_frontend", br / "perch_v2_spectrogram.pth"))
        bw_p = Path(self.config.get("distill_perch_pytorch_backbone", br / "perch_v2_backbone.effb3.pth"))
        self.perch_extractor = HybridPerchExtractor(
            onnx_path=onnx_p,
            frontend_weights=fe_p,
            backbone_weights=bw_p,
            n_fixed_samples=int(self.config.get("distill_perch_fixed_samples", 160000)),
            lazy_torch=bool(self.config.get("distill_perch_lazy_torch", True)),
        )
        self.perch_extractor.eval()
        self.distill_perch_coef = float(self.config.get("distill_perch_coef", 1.0))
        self.distill_perch_emb_loss_alpha = float(self.config.get("distill_perch_emb_loss_alpha", 0.5))
        self.distill_perch_do_norm = bool(self.config.get("distill_perch_do_norm", True))
        init_layer(self.distill_head)

    def apply_online_augmentations(self, x, y):
        return x, y

    def forward(self, x, y=None, perch_wave=None, *, skip_online_augmentations=False):
        _ = y, skip_online_augmentations
        feats = self.encoder(x)
        student_emb = self.distill_head(feats)
        loss = None
        if perch_wave is not None:
            with torch.no_grad():
                teacher_emb = self.perch_extractor(perch_wave)
            loss = perch_embedding_distill_loss(
                student_emb,
                teacher_emb,
                alpha=self.distill_perch_emb_loss_alpha,
                coef=self.distill_perch_coef,
                do_norm=self.distill_perch_do_norm,
            )
        logits = student_emb
        probs = torch.sigmoid(logits)
        return {
            "framewise_output": None,
            "framewise_logit": None,
            "segmentwise_output": None,
            "clipwise_output": probs,
            "clipwise_logit": logits,
            "logit": logits,
            "distill_loss": None,
            "loss": loss,
        }
