import math
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.blocks import (
    AttBlockV2,
    GeMFreqPool,
    attn_block_kw_from_cfg,
    init_bn,
    init_layer,
    interpolate,
    pad_framewise_output,
)
from src.dataset import reshape_waveform
from src.losses import build_loss
from src.melspec import AudioToSpec
from src.online_augmentations import (
    horizontal_cutmix_mel,
    horizontal_cutmix_stacked_mel,
    mixup,
    mixup_stacked_mel,
    sumix_freq,
    sumix_freq_stacked_mel,
)
from src.perch import HybridPerchExtractor, load_perchv2_pytorch
from src.perch_distill_loss import perch_embedding_distill_loss
from src.proto_pnet_head import ProtoPNetHeadTorch
from src.ss_soundscape_bank import build_soundscape_augmentation_bank


def get_timm_model(backbone_config, *, num_channels: int = 3):
    return timm.create_model(
        backbone_config["backbone_name"],
        pretrained=backbone_config["pretrained"],
        pretrained_cfg_overlay={"file": f"{backbone_config['backbone_name']}/model.safetensors"},
        drop_rate=backbone_config["drop_rate"],
        drop_path_rate=backbone_config["drop_path_rate"],
        in_chans=int(num_channels),
    )


def timm_efficientnet_to_sequential_encoder_state_dict(sd):
    out = {}
    for k, v in sd.items():
        if k.startswith("conv_stem."):
            out["0." + k.removeprefix("conv_stem.")] = v
        elif k.startswith("bn1."):
            out["1." + k.removeprefix("bn1.")] = v
        elif k.startswith("blocks."):
            out["2." + k.removeprefix("blocks.")] = v
        elif k.startswith("conv_head."):
            out["3." + k.removeprefix("conv_head.")] = v
        elif k.startswith("bn2."):
            out["4." + k.removeprefix("bn2.")] = v
    return out


def timm_convnext_native_to_sequential_encoder_state_dict(sd: dict) -> dict:
    out: dict = {}
    for k, v in sd.items():
        sk = str(k)
        if sk.startswith("stem."):
            out["0." + sk.removeprefix("stem.")] = v
        elif sk.startswith("stages."):
            out["1." + sk.removeprefix("stages.")] = v
    return out


def timm_eca_nfnet_native_to_sequential_encoder_state_dict(sd: dict) -> dict:
    out: dict = {}
    for k, v in sd.items():
        sk = str(k)
        if sk.startswith("stem."):
            out["0." + sk.removeprefix("stem.")] = v
        elif sk.startswith("stages."):
            out["1." + sk.removeprefix("stages.")] = v
        elif sk.startswith("final_conv."):
            out["2." + sk.removeprefix("final_conv.")] = v
    return out


def timm_regnety_native_to_sequential_encoder_state_dict(sd: dict) -> dict:
    out: dict = {}
    for k, v in sd.items():
        sk = str(k)
        if sk.startswith("stem."):
            out["0." + sk.removeprefix("stem.")] = v
        elif sk.startswith("s1."):
            out["1." + sk.removeprefix("s1.")] = v
        elif sk.startswith("s2."):
            out["2." + sk.removeprefix("s2.")] = v
        elif sk.startswith("s3."):
            out["3." + sk.removeprefix("s3.")] = v
        elif sk.startswith("s4."):
            out["4." + sk.removeprefix("s4.")] = v
    return out


def timm_resnet_stem_layers_native_to_sequential_encoder_state_dict(sd: dict) -> dict:
    out: dict = {}
    prefixes = (
        ("conv1.", "0."),
        ("bn1.", "1."),
        ("layer1.", "4."),
        ("layer2.", "5."),
        ("layer3.", "6."),
        ("layer4.", "7."),
    )
    for k, v in sd.items():
        sk = str(k)
        for old_p, new_p in prefixes:
            if sk.startswith(old_p):
                out[new_p + sk.removeprefix(old_p)] = v
                break
    return out


def prepare_timm_efficientnet_encoder_state_dict(raw_sd: dict, *, target_in_chans: int = 3) -> dict:
    sd = dict(raw_sd)
    if "model_state_dict" in sd and isinstance(sd["model_state_dict"], dict):
        sd = dict(sd["model_state_dict"])
    if any(str(k).startswith("module.encoder.") for k in sd):
        sd = {str(k).removeprefix("module.encoder."): v for k, v in sd.items() if str(k).startswith("module.encoder.")}
    elif any(str(k).startswith("encoder.") for k in sd):
        sd = {str(k).removeprefix("encoder."): v for k, v in sd.items() if str(k).startswith("encoder.")}
    if not sd:
        return sd
    has_sequential_idx = any(str(k).startswith(("0.", "1.", "2.", "3.", "4.")) for k in sd)
    has_eff_raw_keys = any(
        str(k).startswith(p) for k in sd for p in ("conv_stem.", "blocks.", "conv_head.", "bn2.")
    )
    has_eca_nfnet_native = any(str(k).startswith("final_conv.") for k in sd)
    has_convnext_native = (
        not has_eca_nfnet_native
        and any(str(k).startswith("stem.") for k in sd)
        and any(str(k).startswith("stages.") for k in sd)
    )
    has_regnety_native = any(str(k).startswith("stem.") for k in sd) and any(
        str(k).startswith("s1.") for k in sd
    )
    has_resnet_trunk_native = any(str(k).startswith("conv1.") for k in sd) and any(
        str(k).startswith("layer1.") for k in sd
    )
    if not has_sequential_idx and has_eff_raw_keys:
        sd = timm_efficientnet_to_sequential_encoder_state_dict(sd)
    elif not has_sequential_idx and has_eca_nfnet_native:
        sd = timm_eca_nfnet_native_to_sequential_encoder_state_dict(sd)
    elif not has_sequential_idx and has_convnext_native:
        sd = timm_convnext_native_to_sequential_encoder_state_dict(sd)
    elif not has_sequential_idx and has_regnety_native:
        sd = timm_regnety_native_to_sequential_encoder_state_dict(sd)
    elif not has_sequential_idx and has_resnet_trunk_native:
        sd = timm_resnet_stem_layers_native_to_sequential_encoder_state_dict(sd)
    for key in ("0.weight", "0.0.weight", "conv_stem.weight"):
        if key not in sd:
            continue
        w = sd[key]
        if w.dim() == 4:
            cin = int(w.shape[1])
            if cin == 1 and int(target_in_chans) == 3:
                sd[key] = w.expand(-1, 3, -1, -1).clone() / 3.0
            elif cin == 3 and int(target_in_chans) == 1:
                sd[key] = w.mean(dim=1, keepdim=True)
            break
    return sd


def _sed_timm_backbone_key(backbone_name: str) -> str:
    return Path(backbone_name).name.lower()


def sed_timm_encoder_layers_and_in_features(base_model: nn.Module, backbone_name: str) -> tuple[list[nn.Module], int]:
    k = _sed_timm_backbone_key(backbone_name)
    ch = list(base_model.children())
    if "hgnet" in k:
        return ch[:-1], int(base_model.num_features)
    layers = ch[:-2]
    if (
        "efficientnetv2" in k
        or "tf_efficientnetv2" in k
        or "efficientnet" in k
        or k.startswith("tf_efficientnet")
    ):
        in_features = int(base_model.classifier.in_features)
    elif "eca" in k:
        in_features = int(base_model.head.fc.in_features)
    elif "convnext" in k:
        in_features = int(base_model.head.fc.in_features)
    elif "regnety" in k or "regnet" in k:
        in_features = int(base_model.head.fc.in_features)
    elif "spnasnet" in k:
        in_features = int(base_model.classifier.in_features)
    elif "res" in k:
        in_features = int(base_model.fc.in_features)
    else:
        raise NotImplementedError(f"SED timm backbone not supported for in_features: {backbone_name!r}")
    return layers, in_features


def multicontext_k_from_cfg(cfg: dict) -> int:
    cd = float(cfg["dataset"]["chunk_duration_s"])
    bs = float(cfg["dataset"].get("soundscape_label_bin_s", 5.0))
    if bs <= 0:
        return 1
    return max(1, round(cd / bs))


def encoder_bchw_to_ct(
    x: torch.Tensor,
    channel_smoothing: str,
    gem_freq_pool,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    if channel_smoothing == "gemfreq":
        x = gem_freq_pool(x)
    elif channel_smoothing == "max_plus_avg":
        x = torch.mean(x, dim=2)
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
    else:
        raise NotImplementedError
    x = F.dropout(x, p=dropout_p, training=training)
    return x


def split_encoder_bchw_along_time(x: torch.Tensor, k: int) -> list[torch.Tensor]:
    tw = x.size(3)
    edges = torch.linspace(0, tw, steps=max(2, int(k) + 1), device=x.device).round().long()
    chunks = []
    for i in range(int(k)):
        l = int(edges[i].item())
        r = int(edges[i + 1].item())
        if r <= l:
            r = min(l + 1, tw)
        r = min(r, tw)
        if l >= r:
            l, r = max(0, tw - 1), tw
        chunks.append(x[:, :, :, l:r])
    return chunks


def segmentwise_logit_to_agg(segmentwise_logit: torch.Tensor, pooling: str) -> torch.Tensor:
    if pooling == "max":
        return segmentwise_logit.max(dim=1)[0]
    if pooling == "mean":
        return segmentwise_logit.mean(dim=1)
    if pooling == "lse":
        r = 10.0
        return torch.logsumexp(r * segmentwise_logit, dim=1) / r
    if pooling == "gem":
        p = 3.0
        z = segmentwise_logit.transpose(1, 2)
        return F.avg_pool1d(z.pow(p), kernel_size=z.size(2)).pow(1.0 / p).view(z.size(0), -1)
    if pooling == "noisy_or":
        probs = torch.sigmoid(segmentwise_logit)
        eps = 1e-7
        bag_probs = 1 - torch.prod(1 - probs + eps, dim=1)
        return torch.logit(bag_probs, eps=eps)
    if pooling == "topk":
        kk = 3
        topk_vals, _ = segmentwise_logit.topk(kk, dim=1)
        return topk_vals.mean(dim=1)
    raise NotImplementedError


def sinusoidal_pe_length(length: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def multilabel_bernoulli_kl_mean(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    q = q.clamp(min=eps, max=1.0 - eps)
    return (p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())).mean()


def _use_top4_input_bn_freq_axis(config: dict) -> bool:
    model_flag = bool(config.get("model", {}).get("use_top4_input_bn_freq_axis", False))
    mel_flag = bool(config.get("mel_spec_params", {}).get("use_top4_method", False))
    return model_flag or mel_flag


def _top4_bn0_num_features(config: dict) -> int:
    msp = config.get("mel_spec_params", {})
    image_size = msp.get("mel_image_size")
    if image_size is not None:
        return int(image_size[0])  # (H, W), top4 bn0 is over mel-height/freq axis
    return int(msp.get("n_mels", 128))


def apply_input_bn(x: torch.Tensor, bn0: nn.BatchNorm2d, *, top4_freq_axis: bool) -> torch.Tensor:
    if top4_freq_axis:
        # x (B, 1, H, W) -> transpose channel<->freq -> BN2d(H) -> transpose back
        x = x.transpose(1, 2)
        x = bn0(x)
        x = x.transpose(1, 2)
        return x
    return bn0(x)


class SEDModelClassic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_chans = int(self.config["model"].get("num_channels", 3))
        self.use_top4_input_bn_freq_axis = _use_top4_input_bn_freq_axis(self.config)
        bn0_features = _top4_bn0_num_features(self.config) if self.use_top4_input_bn_freq_axis else self.input_chans
        self.bn0 = nn.BatchNorm2d(bn0_features)

        base_model = get_timm_model(self.config["model"]["backbone"], num_channels=self.input_chans)
        enc_layers, in_features = sed_timm_encoder_layers_and_in_features(
            base_model, self.config["model"]["backbone"]["backbone_name"]
        )
        self.encoder = nn.Sequential(*enc_layers)

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        ab = self.config["model"]["attn_block"]
        self.att_block = AttBlockV2(
            in_features,
            self.config["num_classes"],
            **attn_block_kw_from_cfg(ab),
        )
        self.segwise_pooling = ab["segwise_pooling"]
        self.channel_smoothing = ab["channel_smoothing"]
        self.gem_freq_pool = GeMFreqPool() if self.channel_smoothing == "gemfreq" else None

        self.use_five_head_dropouts = bool(ab.get("use_five_head_dropouts", False))
        if self.use_five_head_dropouts:
            probs = ab.get("five_head_dropout_probs")
            if probs is None:
                probs = torch.linspace(0.1, 0.5, 5).tolist()
            self.head_dropouts = nn.ModuleList([nn.Dropout(float(p)) for p in probs])
        else:
            self.head_dropouts = None

        self.criterion = build_loss(self.config)
        self.distill_perch = bool(self.config.get("distill_perch", False))
        self.distill_perch_coef = float(self.config.get("distill_perch_coef", 1.0))
        self.distill_perch_emb_loss_alpha = float(self.config.get("distill_perch_emb_loss_alpha", 0.5))
        self.distill_perch_do_norm = bool(self.config.get("distill_perch_do_norm", True))
        self.distill_head = None
        self.perch_extractor = None
        if self.distill_perch:
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

        oa = self.config["online_aug"]
        self.wave_level_online_aug = oa["wave_level"]
        self.use_reshape_waves_not_melspec = self.config["dataset"]["use_reshape_waves_not_melspec"]
        self.wave_reshape_width = self.config["dataset"]["wave_reshape_width"]
        self.chunked_multicontext = bool(self.config["model"].get("chunked_multicontext", False))
        self.audio_to_spec = AudioToSpec(config) if self.wave_level_online_aug else None

        self.ss_bank_x = None
        self.ss_bank_y = None

        if oa["use_ss_bank"]:
            bx, by = build_soundscape_augmentation_bank(self.config)
            if bx is not None:
                self.ss_bank_x = bx
                self.ss_bank_y = by
                print(f"[model] soundscape mix bank: {bx.shape[0]} tiles")
            else:
                print("[model] use_ss_bank=True but bank is empty")

        self.use_symm_kl_div = bool(self.config.get("use_symm_kl_div", False))

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        if self.distill_head is not None:
            init_layer(self.distill_head)

        ck = self.config["model"]["backbone"]["init_checkpoint"]
        if ck:
            raw = torch.load(ck, map_location="cpu", weights_only=True)
            sd = prepare_timm_efficientnet_encoder_state_dict(raw, target_in_chans=self.input_chans)
            miss, unexpected = self.encoder.load_state_dict(sd, strict=False)
            if miss or unexpected:
                print(
                    f"[model] encoder init_checkpoint loaded with strict=False: missing={len(miss)} unexpected={len(unexpected)}"
                )
            else:
                print(f"[model] encoder weights loaded from {ck}")

    def apply_online_augmentations(self, x, y):
        aug_cfg = self.config["online_aug"]
        p_mixup, mixup_alpha, p_sumix = aug_cfg["p_mixup"], aug_cfg["mixup_alpha"], aug_cfg["p_sumix_freq"]
        share = aug_cfg["ss_bank_share"]
        use_bank = aug_cfg["use_ss_bank"] and self.ss_bank_x is not None and share > 0.0
        bank_x = self.ss_bank_x if use_bank else None
        bank_y = self.ss_bank_y if use_bank else None
        share_kw = share if use_bank else 0.0
        apply_ss_bank_wave_aug = bool(self.wave_level_online_aug and use_bank)
        wave_aug_cfg = self.config["wave_aug"] if apply_ss_bank_wave_aug else None
        mixup_balancing = bool(aug_cfg.get("mixup_balancing", False)) and share_kw <= 0.0
        mixup_balance_active_eps = float(aug_cfg.get("mixup_balance_active_eps", 0.5))
        if p_mixup > 0.0 and mixup_alpha > 0.0 and torch.rand((), device=x.device) < p_mixup:
            x, y = mixup(
                x,
                y,
                mixup_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        if p_sumix > 0.0 and torch.rand((), device=x.device) < p_sumix:
            x, y = sumix_freq(
                x,
                y,
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
            )
        p_hcm = float(aug_cfg.get("p_horizontal_cutmix", 0.0))
        if p_hcm > 0.0 and not self.wave_level_online_aug and x.dim() == 4 and torch.rand((), device=x.device) < p_hcm:
            hcm_alpha = float(aug_cfg.get("horizontal_cutmix_alpha", 1.0))
            x, y = horizontal_cutmix_mel(
                x,
                y,
                hcm_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        return x, y

    def wave_batch_to_model_input(self, x, training):
        out = []
        for i in range(x.size(0)):
            w = x[i]
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if self.use_reshape_waves_not_melspec:
                out.append(reshape_waveform(w, self.wave_reshape_width))
            else:
                out.append(self.audio_to_spec(w, training))
        return torch.stack(out, dim=0)

    def segmentwise_bin_max_logits(self, segmentwise_logit, k):
        t = segmentwise_logit.size(1)
        edges = torch.linspace(0, t, steps=max(2, int(k) + 1), device=segmentwise_logit.device).round().long()
        per_bin = []
        for i in range(int(k)):
            l = int(edges[i].item())
            r = int(edges[i + 1].item())
            if r <= l:
                continue
            per_bin.append(segmentwise_logit[:, l:r, :].max(dim=1)[0])
        if not per_bin:
            return None
        return torch.stack(per_bin, dim=1)  # (B, k_eff, C)

    def forward(
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
        # x: (batch_size, 3, h, w) mel/reshape from dataset, or (batch_size, 1, t) raw wave at wave_level
        perch_input = None
        if self.wave_level_online_aug:
            if self.training and y is not None and not skip_online_augmentations:
                x, y = self.apply_online_augmentations(x, y)
            if self.distill_perch and y is not None:
                perch_input = x
            x = self.wave_batch_to_model_input(x, self.training)
        else:
            chunked_input = bool(self.chunked_multicontext and x.ndim == 5)
            if self.training and y is not None and not skip_online_augmentations:
                if chunked_input:
                    b, k, c, h, w = x.shape
                    x_merge = x.permute(0, 2, 3, 1, 4).reshape(b, c, h, k * w)
                    x_merge, y = self.apply_online_augmentations(x_merge, y)
                    x = x_merge.reshape(b, c, h, k, w).permute(0, 3, 1, 2, 4)
                else:
                    x, y = self.apply_online_augmentations(x, y)
            if self.distill_perch and y is not None:
                if perch_wave is None:
                    raise ValueError(
                        "cfg.distill_perch=True with wave_level=False requires augmented waveform in batch "
                        "(perch_wave), same wave as used to build the mel (dataset wave_aug path)."
                    )
                perch_input = perch_wave

            if chunked_input:
                b, k, c, h, w = x.shape
                x = x.reshape(b * k, c, h, w)
                x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
                x = self.encoder(x)
                _bb, ce, fe, te = x.shape
                x = x.reshape(b, k, ce, fe, te).permute(0, 2, 3, 1, 4).reshape(b, ce, fe, k * te)
            else:
                x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
                x = self.encoder(x)

        if self.wave_level_online_aug:
            x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
            x = self.encoder(x)
        distill_loss = None
        if self.distill_perch and y is not None:
            with torch.no_grad():
                teacher_emb = self.perch_extractor(perch_input)
            student_emb = self.distill_head(F.adaptive_avg_pool2d(x, output_size=1).flatten(1))
            distill_loss = perch_embedding_distill_loss(
                student_emb,
                teacher_emb,
                alpha=self.distill_perch_emb_loss_alpha,
                coef=1.0,
                do_norm=self.distill_perch_do_norm,
            )
            use_detach = bool(distill_detach_override) if distill_detach_override is not None else True
            # keep sed head gradients from modifying backbone when distilling from perch
            if use_detach:
                x = x.detach()

        # (batch_size, channels, frames)
        if self.channel_smoothing == "gemfreq":
            x = self.gem_freq_pool(x)
        elif self.channel_smoothing == "max_plus_avg":
            x = torch.mean(x, dim=2)
            x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
            x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
            x = x1 + x2
        else:
            raise NotImplementedError

        x = F.dropout(x, p=self.config["model"]["attn_block"]["dropout"], training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)

        if self.use_five_head_dropouts:
            if self.training:
                clipwise_logit_acc = None
                seg_logit_acc = None
                for d in self.head_dropouts:
                    xd = d(x)
                    _, norm_att, _, cla_logit = self.att_block(xd)
                    clipwise_logit_i = torch.sum(norm_att * cla_logit, dim=2)
                    seg_logit_i = cla_logit.transpose(1, 2)
                    clipwise_logit_acc = (
                        clipwise_logit_i if clipwise_logit_acc is None else clipwise_logit_acc + clipwise_logit_i
                    )
                    seg_logit_acc = seg_logit_i if seg_logit_acc is None else seg_logit_acc + seg_logit_i
                n = len(self.head_dropouts)
                clipwise_logit = clipwise_logit_acc / n
                segmentwise_logit = seg_logit_acc / n
            else:
                _, norm_att, _, cla_logit = self.att_block(x)
                clipwise_logit = torch.sum(norm_att * cla_logit, dim=2)
                segmentwise_logit = cla_logit.transpose(1, 2)
            clipwise_output = torch.sigmoid(clipwise_logit)
            segmentwise_output = torch.sigmoid(segmentwise_logit)
        else:
            x = F.dropout(x, p=self.config["model"]["attn_block"]["dropout"], training=self.training)
            clipwise_output, norm_att, segmentwise_output, cla_logit = self.att_block(x)
            clipwise_logit = torch.sum(norm_att * cla_logit, dim=2)
            segmentwise_logit = cla_logit.transpose(1, 2)
            segmentwise_output = segmentwise_output.transpose(1, 2)

        frames_num = x.size(2)
        interpolate_ratio = frames_num // segmentwise_output.size(1)

        # get framewise output
        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        pooling = self.segwise_pooling
        if pooling == "max":
            agg_logit = segmentwise_logit.max(dim=1)[0]
        elif pooling == "mean":
            agg_logit = segmentwise_logit.mean(dim=1)
        elif pooling == "lse":
            r = 10.0
            agg_logit = torch.logsumexp(r * segmentwise_logit, dim=1) / r
        elif pooling == "gem":
            p = 3.0
            x = segmentwise_logit.transpose(1, 2)
            agg_logit = F.avg_pool1d(x.pow(p), kernel_size=x.size(2)).pow(1.0 / p).view(x.size(0), -1)
        elif pooling == "noisy_or":
            probs = torch.sigmoid(segmentwise_logit)
            eps = 1e-7
            bag_probs = 1 - torch.prod(1 - probs + eps, dim=1)
            agg_logit = torch.logit(bag_probs, eps=eps)
        elif pooling == "topk":
            k = 3
            topk_vals, _ = segmentwise_logit.topk(k, dim=1)
            agg_logit = topk_vals.mean(dim=1)
        else:
            raise NotImplementedError

        if y is not None:
            if (
                (
                    bool(self.config["model"].get("multicontext", False))
                    or bool(self.config["model"].get("chunked_multicontext", False))
                )
                and y_bins is not None
                and y_bins.ndim == 3
                and y_bins.size(1) > 0
            ):
                agg_loss = self.criterion(agg_logit, y)
                per_bin_logits = self.segmentwise_bin_max_logits(segmentwise_logit, y_bins.size(1))
                if per_bin_logits is not None:
                    k_eff = min(per_bin_logits.size(1), y_bins.size(1))
                    if k_eff > 0:
                        per_bin_loss = []
                        for i in range(k_eff):
                            per_bin_loss.append(self.criterion(per_bin_logits[:, i, :], y_bins[:, i, :]))
                        agg_loss = torch.stack(per_bin_loss).mean()
                main_loss = 0.5 * self.criterion(clipwise_logit, y) + 0.5 * agg_loss
                # main_loss = agg_loss
            else:
                main_loss = 0.5 * self.criterion(clipwise_logit, y) + 0.5 * self.criterion(agg_logit, y)
                if self.use_symm_kl_div and self.training and not bool(self.config["model"].get("multicontext", False)):
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
            if distill_loss is not None:
                coef = float(distill_coef_override) if distill_coef_override is not None else self.distill_perch_coef
                loss = main_loss + coef * distill_loss
            else:
                loss = main_loss
        else:
            loss = None

        return {
            "framewise_output": framewise_output,
            "framewise_logit": framewise_logit,
            "segmentwise_output": segmentwise_output,
            "segmentwise_logit": segmentwise_logit,
            "clipwise_output": clipwise_output,
            "clipwise_logit": clipwise_logit,
            "agg_output": torch.sigmoid(agg_logit),
            "agg_logit": agg_logit,
            "logit": clipwise_logit,
            "distill_loss": distill_loss,
            "loss": loss,
        }


class SEDModelProtoSED(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_chans = int(self.config["model"].get("num_channels", 3))
        self.use_top4_input_bn_freq_axis = _use_top4_input_bn_freq_axis(self.config)
        bn0_features = _top4_bn0_num_features(self.config) if self.use_top4_input_bn_freq_axis else self.input_chans
        self.bn0 = nn.BatchNorm2d(bn0_features)

        base_model = get_timm_model(self.config["model"]["backbone"], num_channels=self.input_chans)
        enc_layers, in_features = sed_timm_encoder_layers_and_in_features(
            base_model, self.config["model"]["backbone"]["backbone_name"]
        )
        self.encoder = nn.Sequential(*enc_layers)

        pp = self.config["model"].get("proto_pnet") or {}
        self.proto_num_prototypes = int(pp.get("num_prototypes", 10))
        self.proto_ortho_coef = float(pp.get("ortho_coef", 1.0))
        self.proto_head = ProtoPNetHeadTorch(
            num_classes=int(self.config["num_classes"]),
            embedding_dim=in_features,
            num_prototypes=self.proto_num_prototypes,
            non_negative_kernel=bool(pp.get("non_negative_kernel", True)),
            ortho_loss_weight=float(pp.get("ortho_loss_weight", 1.0)),
            kernel_init_value=float(pp.get("kernel_init_value", 2.0)),
            bias_init_value=float(pp.get("bias_init_value", -2.0)),
            eps=float(pp.get("eps", 1e-5)),
        )

        self.criterion = build_loss(self.config)

        oa = self.config["online_aug"]
        self.wave_level_online_aug = oa["wave_level"]
        self.use_reshape_waves_not_melspec = self.config["dataset"]["use_reshape_waves_not_melspec"]
        self.wave_reshape_width = self.config["dataset"]["wave_reshape_width"]
        self.audio_to_spec = AudioToSpec(config) if self.wave_level_online_aug else None

        self.ss_bank_x = None
        self.ss_bank_y = None
        if oa["use_ss_bank"]:
            bx, by = build_soundscape_augmentation_bank(self.config)
            if bx is not None:
                self.ss_bank_x = bx
                self.ss_bank_y = by
                print(f"[model] soundscape mix bank: {bx.shape[0]} tiles")
            else:
                print("[model] use_ss_bank=True but bank is empty")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        ck = self.config["model"]["backbone"]["init_checkpoint"]
        if ck:
            raw = torch.load(ck, map_location="cpu", weights_only=True)
            sd = prepare_timm_efficientnet_encoder_state_dict(raw, target_in_chans=self.input_chans)
            miss, unexpected = self.encoder.load_state_dict(sd, strict=False)
            if miss or unexpected:
                print(
                    f"[model] encoder init_checkpoint loaded with strict=False: missing={len(miss)} unexpected={len(unexpected)}"
                )
            else:
                print(f"[model] encoder weights loaded from {ck}")

    def apply_online_augmentations(self, x, y):
        aug_cfg = self.config["online_aug"]
        p_mixup, mixup_alpha, p_sumix = aug_cfg["p_mixup"], aug_cfg["mixup_alpha"], aug_cfg["p_sumix_freq"]
        share = aug_cfg["ss_bank_share"]
        use_bank = aug_cfg["use_ss_bank"] and self.ss_bank_x is not None and share > 0.0
        bank_x = self.ss_bank_x if use_bank else None
        bank_y = self.ss_bank_y if use_bank else None
        share_kw = share if use_bank else 0.0
        apply_ss_bank_wave_aug = bool(self.wave_level_online_aug and use_bank)
        wave_aug_cfg = self.config["wave_aug"] if apply_ss_bank_wave_aug else None
        mixup_balancing = bool(aug_cfg.get("mixup_balancing", False)) and share_kw <= 0.0
        mixup_balance_active_eps = float(aug_cfg.get("mixup_balance_active_eps", 0.5))
        if p_mixup > 0.0 and mixup_alpha > 0.0 and torch.rand((), device=x.device) < p_mixup:
            x, y = mixup(
                x,
                y,
                mixup_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        if p_sumix > 0.0 and torch.rand((), device=x.device) < p_sumix:
            x, y = sumix_freq(
                x,
                y,
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
            )
        p_hcm = float(aug_cfg.get("p_horizontal_cutmix", 0.0))
        if p_hcm > 0.0 and not self.wave_level_online_aug and x.dim() == 4 and torch.rand((), device=x.device) < p_hcm:
            hcm_alpha = float(aug_cfg.get("horizontal_cutmix_alpha", 1.0))
            x, y = horizontal_cutmix_mel(
                x,
                y,
                hcm_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        return x, y

    def wave_batch_to_model_input(self, x, training):
        out = []
        for i in range(x.size(0)):
            w = x[i]
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if self.use_reshape_waves_not_melspec:
                out.append(reshape_waveform(w, self.wave_reshape_width))
            else:
                out.append(self.audio_to_spec(w, training))
        return torch.stack(out, dim=0)

    def forward(
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
        if self.wave_level_online_aug:
            if self.training and y is not None and not skip_online_augmentations:
                x, y = self.apply_online_augmentations(x, y)
            x = self.wave_batch_to_model_input(x, self.training)
            x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
            x = self.encoder(x)
        else:
            if x.ndim == 5:
                raise ValueError(
                    "proto_sed does not support multicontext: expected mel (B, C, H, W), got 5D input"
                )
            if self.training and y is not None and not skip_online_augmentations:
                x, y = self.apply_online_augmentations(x, y)
            x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
            x = self.encoder(x)

        proto_in = x.permute(0, 2, 3, 1).contiguous()
        clipwise_logit, ortho_loss = self.proto_head(proto_in)
        clipwise_output = torch.sigmoid(clipwise_logit)

        te = int(x.shape[-1])
        segmentwise_logit = clipwise_logit.unsqueeze(1)
        segmentwise_output = torch.sigmoid(segmentwise_logit)
        framewise_logit = clipwise_logit.unsqueeze(1).expand(-1, te, -1).contiguous()
        framewise_output = torch.sigmoid(framewise_logit)
        agg_logit = clipwise_logit
        agg_output = clipwise_output

        distill_loss = None
        if y is not None:
            main_loss = self.criterion(clipwise_logit, y)
            if self.training:
                main_loss = main_loss + self.proto_ortho_coef * ortho_loss
            loss = main_loss
        else:
            loss = None

        return {
            "framewise_output": framewise_output,
            "framewise_logit": framewise_logit,
            "segmentwise_output": segmentwise_output,
            "segmentwise_logit": segmentwise_logit,
            "clipwise_output": clipwise_output,
            "clipwise_logit": clipwise_logit,
            "agg_output": agg_output,
            "agg_logit": agg_logit,
            "logit": clipwise_logit,
            "distill_loss": distill_loss,
            "loss": loss,
            "proto_ortho_loss": ortho_loss.detach(),
        }


class SEDModelCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_chans = int(self.config["model"].get("num_channels", 3))
        self.use_top4_input_bn_freq_axis = _use_top4_input_bn_freq_axis(self.config)
        bn0_features = _top4_bn0_num_features(self.config) if self.use_top4_input_bn_freq_axis else self.input_chans
        self.bn0 = nn.BatchNorm2d(bn0_features)

        base_model = get_timm_model(self.config["model"]["backbone"], num_channels=self.input_chans)
        enc_layers, in_features = sed_timm_encoder_layers_and_in_features(
            base_model, self.config["model"]["backbone"]["backbone_name"]
        )
        self.encoder = nn.Sequential(*enc_layers)

        self.head_dropout = float(self.config["model"]["attn_block"]["dropout"])
        self.head = nn.Linear(in_features, self.config["num_classes"], bias=True)
        self.criterion = build_loss(self.config)
        self.distill_perch = bool(self.config.get("distill_perch", False))
        self.distill_perch_coef = float(self.config.get("distill_perch_coef", 1.0))
        self.distill_perch_emb_loss_alpha = float(self.config.get("distill_perch_emb_loss_alpha", 0.5))
        self.distill_perch_do_norm = bool(self.config.get("distill_perch_do_norm", True))
        self.distill_head = None
        self.perch_extractor = None
        if self.distill_perch:
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

        oa = self.config["online_aug"]
        self.wave_level_online_aug = oa["wave_level"]
        self.use_reshape_waves_not_melspec = self.config["dataset"]["use_reshape_waves_not_melspec"]
        self.wave_reshape_width = self.config["dataset"]["wave_reshape_width"]
        self.chunked_multicontext = bool(self.config["model"].get("chunked_multicontext", False))
        self.audio_to_spec = AudioToSpec(config) if self.wave_level_online_aug else None

        self.ss_bank_x = None
        self.ss_bank_y = None
        if oa["use_ss_bank"]:
            bx, by = build_soundscape_augmentation_bank(self.config)
            if bx is not None:
                self.ss_bank_x = bx
                self.ss_bank_y = by
                print(f"[model] soundscape mix bank: {bx.shape[0]} tiles")
            else:
                print("[model] use_ss_bank=True but bank is empty")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.head)
        if self.distill_head is not None:
            init_layer(self.distill_head)
        ck = self.config["model"]["backbone"]["init_checkpoint"]
        if ck:
            raw = torch.load(ck, map_location="cpu", weights_only=True)
            sd = prepare_timm_efficientnet_encoder_state_dict(raw, target_in_chans=self.input_chans)
            miss, unexpected = self.encoder.load_state_dict(sd, strict=False)
            if miss or unexpected:
                print(
                    f"[model] encoder init_checkpoint loaded with strict=False: missing={len(miss)} unexpected={len(unexpected)}"
                )
            else:
                print(f"[model] encoder weights loaded from {ck}")

    def apply_online_augmentations(self, x, y):
        aug_cfg = self.config["online_aug"]
        p_mixup, mixup_alpha, p_sumix = aug_cfg["p_mixup"], aug_cfg["mixup_alpha"], aug_cfg["p_sumix_freq"]
        share = aug_cfg["ss_bank_share"]
        use_bank = aug_cfg["use_ss_bank"] and self.ss_bank_x is not None and share > 0.0
        bank_x = self.ss_bank_x if use_bank else None
        bank_y = self.ss_bank_y if use_bank else None
        share_kw = share if use_bank else 0.0
        apply_ss_bank_wave_aug = bool(self.wave_level_online_aug and use_bank)
        wave_aug_cfg = self.config["wave_aug"] if apply_ss_bank_wave_aug else None
        mixup_balancing = bool(aug_cfg.get("mixup_balancing", False)) and share_kw <= 0.0
        mixup_balance_active_eps = float(aug_cfg.get("mixup_balance_active_eps", 0.5))
        if p_mixup > 0.0 and mixup_alpha > 0.0 and torch.rand((), device=x.device) < p_mixup:
            x, y = mixup(
                x,
                y,
                mixup_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        if p_sumix > 0.0 and torch.rand((), device=x.device) < p_sumix:
            x, y = sumix_freq(
                x,
                y,
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
            )
        p_hcm = float(aug_cfg.get("p_horizontal_cutmix", 0.0))
        if p_hcm > 0.0 and not self.wave_level_online_aug and x.dim() == 4 and torch.rand((), device=x.device) < p_hcm:
            hcm_alpha = float(aug_cfg.get("horizontal_cutmix_alpha", 1.0))
            x, y = horizontal_cutmix_mel(
                x,
                y,
                hcm_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        return x, y

    def wave_batch_to_model_input(self, x, training):
        out = []
        for i in range(x.size(0)):
            w = x[i]
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if self.use_reshape_waves_not_melspec:
                out.append(reshape_waveform(w, self.wave_reshape_width))
            else:
                out.append(self.audio_to_spec(w, training))
        return torch.stack(out, dim=0)

    def forward(
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
        _ = y_bins
        perch_input = None
        if self.wave_level_online_aug:
            if self.training and y is not None and not skip_online_augmentations:
                x, y = self.apply_online_augmentations(x, y)
            if self.distill_perch and y is not None:
                perch_input = x
            x = self.wave_batch_to_model_input(x, self.training)
            x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
            x = self.encoder(x)
        else:
            chunked_input = bool(self.chunked_multicontext and x.ndim == 5)
            if self.training and y is not None and not skip_online_augmentations:
                if chunked_input:
                    b, k, c, h, w = x.shape
                    x_merge = x.permute(0, 2, 3, 1, 4).reshape(b, c, h, k * w)
                    x_merge, y = self.apply_online_augmentations(x_merge, y)
                    x = x_merge.reshape(b, c, h, k, w).permute(0, 3, 1, 2, 4)
                else:
                    x, y = self.apply_online_augmentations(x, y)
            if self.distill_perch and y is not None:
                if perch_wave is None:
                    raise ValueError(
                        "cfg.distill_perch=True with wave_level=False requires augmented waveform in batch "
                        "(perch_wave), same wave as used to build the mel (dataset wave_aug path)."
                    )
                perch_input = perch_wave

            if chunked_input:
                b, k, c, h, w = x.shape
                x = x.reshape(b * k, c, h, w)
                x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
                x = self.encoder(x)
                _, ce, fe, te = x.shape
                x = x.reshape(b, k, ce, fe, te).permute(0, 2, 3, 1, 4).reshape(b, ce, fe, k * te)
            else:
                x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
                x = self.encoder(x)

        distill_loss = None
        if self.distill_perch and y is not None:
            with torch.no_grad():
                teacher_emb = self.perch_extractor(perch_input)
            student_emb = self.distill_head(F.adaptive_avg_pool2d(x, output_size=1).flatten(1))
            distill_loss = perch_embedding_distill_loss(
                student_emb,
                teacher_emb,
                alpha=self.distill_perch_emb_loss_alpha,
                coef=1.0,
                do_norm=self.distill_perch_do_norm,
            )
            use_detach = bool(distill_detach_override) if distill_detach_override is not None else True
            if use_detach:
                x = x.detach()

        feats = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        feats = F.dropout(feats, p=self.head_dropout, training=self.training)
        clipwise_logit = self.head(feats)
        clipwise_output = torch.sigmoid(clipwise_logit)
        agg_logit = clipwise_logit

        if y is not None:
            main_loss = self.criterion(clipwise_logit, y)
            if distill_loss is not None:
                coef = float(distill_coef_override) if distill_coef_override is not None else self.distill_perch_coef
                loss = main_loss + coef * distill_loss
            else:
                loss = main_loss
        else:
            loss = None

        return {
            "framewise_output": None,
            "framewise_logit": None,
            "segmentwise_output": None,
            "segmentwise_logit": None,
            "clipwise_output": clipwise_output,
            "clipwise_logit": clipwise_logit,
            "agg_output": clipwise_output,
            "agg_logit": agg_logit,
            "logit": clipwise_logit,
            "distill_loss": distill_loss,
            "loss": loss,
        }


class SEDModelMultiKHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if not bool(config["model"].get("multicontext")):
            raise ValueError("model_type=multised requires cfg.model.multicontext=True")
        self.k = multicontext_k_from_cfg(config)

        self.input_chans = int(self.config["model"].get("num_channels", 3))
        self.use_top4_input_bn_freq_axis = _use_top4_input_bn_freq_axis(self.config)
        bn0_features = _top4_bn0_num_features(self.config) if self.use_top4_input_bn_freq_axis else self.input_chans
        self.bn0 = nn.BatchNorm2d(bn0_features)
        base_model = get_timm_model(self.config["model"]["backbone"], num_channels=self.input_chans)
        enc_layers, in_features = sed_timm_encoder_layers_and_in_features(
            base_model, self.config["model"]["backbone"]["backbone_name"]
        )
        self.encoder = nn.Sequential(*enc_layers)

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        ab = self.config["model"]["attn_block"]
        kw = attn_block_kw_from_cfg(ab)
        self.att_blocks = nn.ModuleList(
            [AttBlockV2(in_features, self.config["num_classes"], **kw) for _ in range(self.k)]
        )
        self.segwise_pooling = ab["segwise_pooling"]
        self.channel_smoothing = ab["channel_smoothing"]
        self.gem_freq_pool = GeMFreqPool() if self.channel_smoothing == "gemfreq" else None

        self.criterion = build_loss(self.config)

        oa = self.config["online_aug"]
        self.wave_level_online_aug = oa["wave_level"]
        self.use_reshape_waves_not_melspec = self.config["dataset"]["use_reshape_waves_not_melspec"]
        self.wave_reshape_width = self.config["dataset"]["wave_reshape_width"]
        self.chunked_multicontext = bool(self.config["model"].get("chunked_multicontext", False))
        self.audio_to_spec = AudioToSpec(config) if self.wave_level_online_aug else None

        self.ss_bank_x = None
        self.ss_bank_y = None
        if oa["use_ss_bank"]:
            bx, by = build_soundscape_augmentation_bank(self.config)
            if bx is not None:
                self.ss_bank_x = bx
                self.ss_bank_y = by
                print(f"[model] soundscape mix bank: {bx.shape[0]} tiles")
            else:
                print("[model] use_ss_bank=True but bank is empty")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        ck = self.config["model"]["backbone"]["init_checkpoint"]
        if ck:
            raw = torch.load(ck, map_location="cpu", weights_only=True)
            sd = prepare_timm_efficientnet_encoder_state_dict(raw, target_in_chans=self.input_chans)
            miss, unexpected = self.encoder.load_state_dict(sd, strict=False)
            if miss or unexpected:
                print(
                    f"[model] encoder init_checkpoint loaded with strict=False: missing={len(miss)} unexpected={len(unexpected)}"
                )
            else:
                print(f"[model] encoder weights loaded from {ck}")

    def apply_online_augmentations(self, x, y):
        aug_cfg = self.config["online_aug"]
        p_mixup, mixup_alpha, p_sumix = aug_cfg["p_mixup"], aug_cfg["mixup_alpha"], aug_cfg["p_sumix_freq"]
        share = aug_cfg["ss_bank_share"]
        use_bank = aug_cfg["use_ss_bank"] and self.ss_bank_x is not None and share > 0.0
        bank_x = self.ss_bank_x if use_bank else None
        bank_y = self.ss_bank_y if use_bank else None
        share_kw = share if use_bank else 0.0
        apply_ss_bank_wave_aug = bool(self.wave_level_online_aug and use_bank)
        wave_aug_cfg = self.config["wave_aug"] if apply_ss_bank_wave_aug else None
        mixup_balancing = bool(aug_cfg.get("mixup_balancing", False)) and share_kw <= 0.0
        mixup_balance_active_eps = float(aug_cfg.get("mixup_balance_active_eps", 0.5))
        if p_mixup > 0.0 and mixup_alpha > 0.0 and torch.rand((), device=x.device) < p_mixup:
            x, y = mixup(
                x,
                y,
                mixup_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        if p_sumix > 0.0 and torch.rand((), device=x.device) < p_sumix:
            x, y = sumix_freq(
                x,
                y,
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
            )
        p_hcm = float(aug_cfg.get("p_horizontal_cutmix", 0.0))
        if p_hcm > 0.0 and not self.wave_level_online_aug and x.dim() == 4 and torch.rand((), device=x.device) < p_hcm:
            hcm_alpha = float(aug_cfg.get("horizontal_cutmix_alpha", 1.0))
            x, y = horizontal_cutmix_mel(
                x,
                y,
                hcm_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        return x, y

    def wave_batch_to_model_input(self, x, training):
        out = []
        for i in range(x.size(0)):
            w = x[i]
            if w.dim() == 1:
                w = w.unsqueeze(0)
            if self.use_reshape_waves_not_melspec:
                out.append(reshape_waveform(w, self.wave_reshape_width))
            else:
                out.append(self.audio_to_spec(w, training))
        return torch.stack(out, dim=0)

    def forward(
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
        _ = perch_wave, distill_detach_override, distill_coef_override
        if self.wave_level_online_aug:
            if self.training and y is not None and not skip_online_augmentations:
                x, y = self.apply_online_augmentations(x, y)
            x = self.wave_batch_to_model_input(x, self.training)
        else:
            if self.training and y is not None and not skip_online_augmentations:
                x, y = self.apply_online_augmentations(x, y)

        x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
        x = self.encoder(x)
        drop_p = self.config["model"]["attn_block"]["dropout"]
        chunks = split_encoder_bchw_along_time(x, self.k)

        per_bin_agg = []
        seg_logits = []
        seg_outputs = []
        clip_logits_l = []
        clip_outputs_l = []
        framewise_logits = []
        framewise_outputs = []
        frames_num_total = 0

        for _i, (xc, head) in enumerate(zip(chunks, self.att_blocks, strict=False)):
            seq = encoder_bchw_to_ct(xc, self.channel_smoothing, self.gem_freq_pool, drop_p, self.training)
            frames_num_total += seq.size(2)
            x_seq = seq.transpose(1, 2)
            x_seq = F.relu_(self.fc1(x_seq))
            x_seq = x_seq.transpose(1, 2)
            x_seq = F.dropout(x_seq, p=drop_p, training=self.training)

            clipwise_output, norm_att, segmentwise_output, cla_logit = head(x_seq)
            clipwise_logit_i = torch.sum(norm_att * cla_logit, dim=2)
            segmentwise_logit = cla_logit.transpose(1, 2)
            segmentwise_output = segmentwise_output.transpose(1, 2)

            tlen = seq.size(2)
            interpolate_ratio = max(1, tlen // max(1, segmentwise_output.size(1)))
            fw_o = interpolate(segmentwise_output, interpolate_ratio)
            fw_o = pad_framewise_output(fw_o, tlen)
            fw_l = interpolate(segmentwise_logit, interpolate_ratio)
            fw_l = pad_framewise_output(fw_l, tlen)

            per_bin_agg.append(segmentwise_logit_to_agg(segmentwise_logit, self.segwise_pooling))
            seg_logits.append(segmentwise_logit)
            seg_outputs.append(segmentwise_output)
            clip_logits_l.append(clipwise_logit_i)
            clip_outputs_l.append(clipwise_output)
            framewise_logits.append(fw_l)
            framewise_outputs.append(fw_o)

        segmentwise_logit = torch.cat(seg_logits, dim=1)
        segmentwise_output = torch.cat(seg_outputs, dim=1)
        framewise_logit = torch.cat(framewise_logits, dim=1)
        framewise_output = torch.cat(framewise_outputs, dim=1)
        per_bin_agg_logit = torch.stack(per_bin_agg, dim=1)
        clipwise_logit = torch.stack(clip_logits_l, dim=1).mean(dim=1)
        clipwise_output = torch.stack(clip_outputs_l, dim=1).mean(dim=1)
        agg_logit = per_bin_agg_logit.max(dim=1)[0]

        if y is not None:
            if y_bins is not None and y_bins.ndim == 3 and y_bins.size(1) > 0:
                k_eff = min(per_bin_agg_logit.size(1), y_bins.size(1))
                if k_eff > 0:
                    main_loss = torch.stack(
                        [self.criterion(per_bin_agg_logit[:, i, :], y_bins[:, i, :]) for i in range(k_eff)]
                    ).mean()
                else:
                    main_loss = self.criterion(agg_logit, y)
            else:
                main_loss = 0.5 * self.criterion(clipwise_logit, y) + 0.5 * self.criterion(agg_logit, y)
            loss = main_loss
        else:
            loss = None

        return {
            "framewise_output": framewise_output,
            "framewise_logit": framewise_logit,
            "segmentwise_output": segmentwise_output,
            "segmentwise_logit": segmentwise_logit,
            "clipwise_output": clipwise_output,
            "clipwise_logit": clipwise_logit,
            "agg_output": torch.sigmoid(agg_logit),
            "agg_logit": agg_logit,
            "per_bin_agg_logit": per_bin_agg_logit,
            "logit": clipwise_logit,
            "distill_loss": None,
            "loss": loss,
        }


class SEDModelMultiTrans(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if not bool(config["model"].get("multicontext")):
            raise ValueError("model_type=multised_trans requires cfg.model.multicontext=True")
        self.k = multicontext_k_from_cfg(config)

        self.input_chans = int(self.config["model"].get("num_channels", 3))
        self.use_top4_input_bn_freq_axis = _use_top4_input_bn_freq_axis(self.config)
        bn0_features = _top4_bn0_num_features(self.config) if self.use_top4_input_bn_freq_axis else self.input_chans
        self.bn0 = nn.BatchNorm2d(bn0_features)
        base_model = get_timm_model(self.config["model"]["backbone"], num_channels=self.input_chans)
        enc_layers, in_features = sed_timm_encoder_layers_and_in_features(
            base_model, self.config["model"]["backbone"]["backbone_name"]
        )
        self.encoder = nn.Sequential(*enc_layers)

        tt = str(config["model"].get("multised_trans_temporal", "gru")).lower()
        if tt not in ("gru", "transformer"):
            raise ValueError("cfg.model.multised_trans_temporal must be 'gru' or 'transformer'")
        self.temporal_type = tt
        if tt == "gru":
            hidden = int(config["model"].get("multised_trans_gru_hidden", in_features))
            n_layers = int(config["model"].get("multised_trans_gru_layers", 1))
            bidir = bool(config["model"].get("multised_trans_gru_bidirectional", False))
            self.temporal = nn.GRU(
                in_features,
                hidden,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=bidir,
            )
            out_h = hidden * (2 if bidir else 1)
            self.temporal_proj = nn.Linear(out_h, in_features) if out_h != in_features else nn.Identity()
        else:
            self.temporal_proj = nn.Identity()
            nhead = int(config["model"].get("multised_trans_nhead", 8))
            if in_features % nhead != 0:
                raise ValueError(f"backbone in_features={in_features} not divisible by multised_trans_nhead={nhead}")
            dim_ff = int(config["model"].get("multised_trans_ffn", max(512, in_features * 2)))
            nl = int(config["model"].get("multised_trans_layers", 2))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=in_features,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=self.config["model"]["attn_block"]["dropout"],
                batch_first=True,
                activation="gelu",
            )
            self.temporal = nn.TransformerEncoder(enc_layer, num_layers=nl)
            self.register_buffer("pos_embed", sinusoidal_pe_length(self.k, in_features), persistent=False)

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        ab = self.config["model"]["attn_block"]
        kw = attn_block_kw_from_cfg(ab)
        self.att_blocks = nn.ModuleList(
            [AttBlockV2(in_features, self.config["num_classes"], **kw) for _ in range(self.k)]
        )
        self.segwise_pooling = ab["segwise_pooling"]
        self.channel_smoothing = ab["channel_smoothing"]
        self.gem_freq_pool = GeMFreqPool() if self.channel_smoothing == "gemfreq" else None

        self.criterion = build_loss(self.config)

        oa = self.config["online_aug"]
        if oa["wave_level"]:
            raise ValueError("multised_trans expects mel input (k stacked specs); set online_aug.wave_level=False")
        self.wave_level_online_aug = False
        self.use_reshape_waves_not_melspec = self.config["dataset"]["use_reshape_waves_not_melspec"]
        if self.use_reshape_waves_not_melspec:
            raise ValueError("multised_trans does not support use_reshape_waves_not_melspec")

        self.ss_bank_x = None
        self.ss_bank_y = None
        if oa["use_ss_bank"]:
            bx, by = build_soundscape_augmentation_bank(self.config)
            if bx is not None:
                self.ss_bank_x = bx
                self.ss_bank_y = by
                print(f"[model] soundscape mix bank: {bx.shape[0]} tiles")
            else:
                print("[model] use_ss_bank=True but bank is empty")

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        if isinstance(self.temporal_proj, nn.Linear):
            init_layer(self.temporal_proj)
        ck = self.config["model"]["backbone"]["init_checkpoint"]
        if ck:
            raw = torch.load(ck, map_location="cpu", weights_only=True)
            sd = prepare_timm_efficientnet_encoder_state_dict(raw, target_in_chans=self.input_chans)
            miss, unexpected = self.encoder.load_state_dict(sd, strict=False)
            if miss or unexpected:
                print(
                    f"[model] encoder init_checkpoint loaded with strict=False: missing={len(miss)} unexpected={len(unexpected)}"
                )
            else:
                print(f"[model] encoder weights loaded from {ck}")

    def apply_online_augmentations(self, x, y, y_bins=None):
        aug_cfg = self.config["online_aug"]
        p_mixup, mixup_alpha, p_sumix = aug_cfg["p_mixup"], aug_cfg["mixup_alpha"], aug_cfg["p_sumix_freq"]
        share = aug_cfg["ss_bank_share"]
        use_bank = aug_cfg["use_ss_bank"] and self.ss_bank_x is not None and share > 0.0
        bank_x = self.ss_bank_x if use_bank else None
        bank_y = self.ss_bank_y if use_bank else None
        share_kw = share if use_bank else 0.0
        mixup_balancing = bool(aug_cfg.get("mixup_balancing", False)) and share_kw <= 0.0
        mixup_balance_active_eps = float(aug_cfg.get("mixup_balance_active_eps", 0.5))
        if x.dim() == 5:
            if use_bank and share_kw > 0.0:
                raise NotImplementedError(
                    "multised_trans: use_ss_bank with ss_bank_share>0 is not supported for stacked mels; "
                    "disable the bank or use model multised / classic sed for bank mixing."
                )
            if p_mixup > 0.0 and mixup_alpha > 0.0 and torch.rand((), device=x.device) < p_mixup:
                x, y, y_bins = mixup_stacked_mel(
                    x,
                    y,
                    y_bins,
                    mixup_alpha,
                    aug_cfg["mixup_use_max_label"],
                    bank_x=None,
                    bank_y=None,
                    ss_bank_share=0.0,
                    apply_ss_bank_wave_aug=False,
                    wave_aug=None,
                    mixup_balancing=mixup_balancing,
                    mixup_balance_active_eps=mixup_balance_active_eps,
                )
            if p_sumix > 0.0 and torch.rand((), device=x.device) < p_sumix:
                x, y, y_bins = sumix_freq_stacked_mel(
                    x,
                    y,
                    y_bins,
                    bank_x=None,
                    bank_y=None,
                    ss_bank_share=0.0,
                    apply_ss_bank_wave_aug=False,
                    wave_aug=None,
                )
            p_hcm = float(aug_cfg.get("p_horizontal_cutmix", 0.0))
            if p_hcm > 0.0 and not self.wave_level_online_aug and torch.rand((), device=x.device) < p_hcm:
                hcm_alpha = float(aug_cfg.get("horizontal_cutmix_alpha", 1.0))
                x, y, y_bins = horizontal_cutmix_stacked_mel(
                    x,
                    y,
                    y_bins,
                    hcm_alpha,
                    aug_cfg["mixup_use_max_label"],
                    bank_x=None,
                    bank_y=None,
                    ss_bank_share=0.0,
                    mixup_balancing=mixup_balancing,
                    mixup_balance_active_eps=mixup_balance_active_eps,
                )
            return x, y, y_bins

        if p_mixup > 0.0 and mixup_alpha > 0.0 and torch.rand((), device=x.device) < p_mixup:
            x, y = mixup(
                x,
                y,
                mixup_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=False,
                wave_aug=None,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        if p_sumix > 0.0 and torch.rand((), device=x.device) < p_sumix:
            x, y = sumix_freq(
                x,
                y,
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=False,
                wave_aug=None,
            )
        p_hcm = float(aug_cfg.get("p_horizontal_cutmix", 0.0))
        if p_hcm > 0.0 and not self.wave_level_online_aug and x.dim() == 4 and torch.rand((), device=x.device) < p_hcm:
            hcm_alpha = float(aug_cfg.get("horizontal_cutmix_alpha", 1.0))
            x, y = horizontal_cutmix_mel(
                x,
                y,
                hcm_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        return x, y, y_bins

    def forward(
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
        _ = perch_wave, distill_detach_override, distill_coef_override
        if x.dim() != 5:
            raise ValueError(f"multised_trans expects x (B,K,C,H,W), got shape {tuple(x.shape)}")
        b, k_, c, h, w = x.shape
        if c != self.input_chans:
            raise ValueError(f"multised_trans input channels C={c} != model.num_channels={self.input_chans}")
        if k_ != self.k:
            raise ValueError(f"Input K={k_} != model k={self.k} (check dataset stacked specs and cfg bins)")

        y_bins_eff = y_bins
        if self.training and y is not None and not skip_online_augmentations:
            x, y, y_bins_eff = self.apply_online_augmentations(x, y, y_bins)

        x = x.view(b * k_, c, h, w)
        x = apply_input_bn(x, self.bn0, top4_freq_axis=self.use_top4_input_bn_freq_axis)
        x = self.encoder(x)
        drop_p = self.config["model"]["attn_block"]["dropout"]
        seq = encoder_bchw_to_ct(x, self.channel_smoothing, self.gem_freq_pool, drop_p, self.training)
        _, c2, t = seq.shape
        seq = seq.view(b, k_, c2, t)

        ctx = seq.mean(dim=-1)
        if self.temporal_type == "gru":
            z, _ = self.temporal(ctx)
            z = self.temporal_proj(z)
        else:
            h = ctx + self.pos_embed.to(device=ctx.device, dtype=ctx.dtype)
            z = self.temporal(h)

        seq = seq + z.unsqueeze(-1)

        per_bin_agg = []
        seg_logits = []
        seg_outputs = []
        clip_logits_l = []
        clip_outputs_l = []
        framewise_logits = []
        framewise_outputs = []

        for i in range(self.k):
            xi = seq[:, i]
            x_seq = xi.transpose(1, 2)
            x_seq = F.relu_(self.fc1(x_seq))
            x_seq = x_seq.transpose(1, 2)
            x_seq = F.dropout(x_seq, p=drop_p, training=self.training)
            clipwise_output, norm_att, segmentwise_output, cla_logit = self.att_blocks[i](x_seq)
            clipwise_logit_i = torch.sum(norm_att * cla_logit, dim=2)
            segmentwise_logit = cla_logit.transpose(1, 2)
            segmentwise_output = segmentwise_output.transpose(1, 2)
            tlen = xi.size(2)
            interpolate_ratio = max(1, tlen // max(1, segmentwise_output.size(1)))
            fw_o = interpolate(segmentwise_output, interpolate_ratio)
            fw_o = pad_framewise_output(fw_o, tlen)
            fw_l = interpolate(segmentwise_logit, interpolate_ratio)
            fw_l = pad_framewise_output(fw_l, tlen)
            per_bin_agg.append(segmentwise_logit_to_agg(segmentwise_logit, self.segwise_pooling))
            seg_logits.append(segmentwise_logit)
            seg_outputs.append(segmentwise_output)
            clip_logits_l.append(clipwise_logit_i)
            clip_outputs_l.append(clipwise_output)
            framewise_logits.append(fw_l)
            framewise_outputs.append(fw_o)

        segmentwise_logit = torch.cat(seg_logits, dim=1)
        segmentwise_output = torch.cat(seg_outputs, dim=1)
        framewise_logit = torch.cat(framewise_logits, dim=1)
        framewise_output = torch.cat(framewise_outputs, dim=1)
        per_bin_agg_logit = torch.stack(per_bin_agg, dim=1)
        clipwise_logit = torch.stack(clip_logits_l, dim=1).mean(dim=1)
        clipwise_output = torch.stack(clip_outputs_l, dim=1).mean(dim=1)
        agg_logit = per_bin_agg_logit.max(dim=1)[0]

        if y is not None:
            if y_bins_eff is not None and y_bins_eff.ndim == 3 and y_bins_eff.size(1) > 0:
                k_eff = min(per_bin_agg_logit.size(1), y_bins_eff.size(1))
                if k_eff > 0:
                    main_loss = torch.stack(
                        [self.criterion(per_bin_agg_logit[:, i, :], y_bins_eff[:, i, :]) for i in range(k_eff)]
                    ).mean()
                else:
                    main_loss = self.criterion(agg_logit, y)
            else:
                main_loss = 0.5 * self.criterion(clipwise_logit, y) + 0.5 * self.criterion(agg_logit, y)
            loss = main_loss
        else:
            loss = None

        return {
            "framewise_output": framewise_output,
            "framewise_logit": framewise_logit,
            "segmentwise_output": segmentwise_output,
            "segmentwise_logit": segmentwise_logit,
            "clipwise_output": clipwise_output,
            "clipwise_logit": clipwise_logit,
            "agg_output": torch.sigmoid(agg_logit),
            "agg_logit": agg_logit,
            "per_bin_agg_logit": per_bin_agg_logit,
            "logit": clipwise_logit,
            "distill_loss": None,
            "loss": loss,
        }


class SEDModelPerch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        pv = load_perchv2_pytorch()
        self.perch_frontend = pv.Perch2Frontend()
        self.perch_backbone = pv.Perch2EfficientB3()

        br = Path(__file__).resolve().parents[1] / "models" / "pytorch_perchv2"
        fe_ckpt = Path(config["model"].get("perch_frontend_ckpt", br / "perch_v2_spectrogram.pth"))
        bw_ckpt = Path(config["model"].get("perch_backbone_ckpt", br / "perch_v2_backbone.effb3.pth"))
        self.perch_frontend.load_state_dict(torch.load(fe_ckpt, map_location="cpu", weights_only=True), strict=True)
        self.perch_backbone.load_state_dict(torch.load(bw_ckpt, map_location="cpu", weights_only=True), strict=True)

        in_features = 1536

        self.fc1 = nn.Linear(in_features, in_features, bias=True)
        ab = self.config["model"]["attn_block"]
        self.att_block = AttBlockV2(
            in_features,
            self.config["num_classes"],
            **attn_block_kw_from_cfg(ab),
        )
        self.segwise_pooling = ab["segwise_pooling"]
        self.channel_smoothing = ab["channel_smoothing"]
        self.gem_freq_pool = GeMFreqPool() if self.channel_smoothing == "gemfreq" else None

        self.criterion = build_loss(self.config)

        oa = self.config["online_aug"]
        if not oa["wave_level"]:
            raise ValueError("SEDModelPerch requires cfg.online_aug.wave_level=True (raw wave -> Perch frontend).")
        self.wave_level_online_aug = True
        self.use_reshape_waves_not_melspec = self.config["dataset"]["use_reshape_waves_not_melspec"]
        self.wave_reshape_width = self.config["dataset"]["wave_reshape_width"]
        self.audio_to_spec = None

        self.ss_bank_x = None
        self.ss_bank_y = None
        if oa["use_ss_bank"]:
            bx, by = build_soundscape_augmentation_bank(self.config)
            if bx is not None:
                self.ss_bank_x = bx
                self.ss_bank_y = by
                print(f"[model] soundscape mix bank: {bx.shape[0]} tiles")
            else:
                print("[model] use_ss_bank=True but bank is empty")

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)

    def apply_online_augmentations(self, x, y):
        aug_cfg = self.config["online_aug"]
        p_mixup, mixup_alpha, p_sumix = aug_cfg["p_mixup"], aug_cfg["mixup_alpha"], aug_cfg["p_sumix_freq"]
        share = aug_cfg["ss_bank_share"]
        use_bank = aug_cfg["use_ss_bank"] and self.ss_bank_x is not None and share > 0.0
        bank_x = self.ss_bank_x if use_bank else None
        bank_y = self.ss_bank_y if use_bank else None
        share_kw = share if use_bank else 0.0
        apply_ss_bank_wave_aug = bool(use_bank)
        wave_aug_cfg = self.config["wave_aug"] if apply_ss_bank_wave_aug else None
        mixup_balancing = bool(aug_cfg.get("mixup_balancing", False)) and share_kw <= 0.0
        mixup_balance_active_eps = float(aug_cfg.get("mixup_balance_active_eps", 0.5))
        if p_mixup > 0.0 and mixup_alpha > 0.0 and torch.rand((), device=x.device) < p_mixup:
            x, y = mixup(
                x,
                y,
                mixup_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        if p_sumix > 0.0 and torch.rand((), device=x.device) < p_sumix:
            x, y = sumix_freq(
                x,
                y,
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                apply_ss_bank_wave_aug=apply_ss_bank_wave_aug,
                wave_aug=wave_aug_cfg,
            )
        p_hcm = float(aug_cfg.get("p_horizontal_cutmix", 0.0))
        if p_hcm > 0.0 and not self.wave_level_online_aug and x.dim() == 4 and torch.rand((), device=x.device) < p_hcm:
            hcm_alpha = float(aug_cfg.get("horizontal_cutmix_alpha", 1.0))
            x, y = horizontal_cutmix_mel(
                x,
                y,
                hcm_alpha,
                aug_cfg["mixup_use_max_label"],
                bank_x=bank_x,
                bank_y=bank_y,
                ss_bank_share=share_kw,
                mixup_balancing=mixup_balancing,
                mixup_balance_active_eps=mixup_balance_active_eps,
            )
        return x, y

    def _encode_perch(self, waves_bt: torch.Tensor) -> torch.Tensor:
        spec = self.perch_frontend(waves_bt)
        x = spec.unsqueeze(1)
        return self.perch_backbone.forward_features(x)

    def wave_batch_to_model_input(self, x, training):
        raise RuntimeError("SEDModelPerch does not use mel inputs; keep wave_level=True.")

    def segmentwise_bin_max_logits(self, segmentwise_logit, k):
        t = segmentwise_logit.size(1)
        edges = torch.linspace(0, t, steps=max(2, int(k) + 1), device=segmentwise_logit.device).round().long()
        per_bin = []
        for i in range(int(k)):
            l = int(edges[i].item())
            r = int(edges[i + 1].item())
            if r <= l:
                continue
            per_bin.append(segmentwise_logit[:, l:r, :].max(dim=1)[0])
        if not per_bin:
            return None
        return torch.stack(per_bin, dim=1)

    def forward(
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
        if self.training and y is not None and not skip_online_augmentations:
            x, y = self.apply_online_augmentations(x, y)

        if x.dim() == 3 and x.size(1) == 1:
            waves = x.squeeze(1)
        else:
            waves = x
        if waves.dim() != 2:
            raise ValueError(f"SEDModelPerch expects wave (B, 1, T) or (B, T), got {tuple(x.shape)}")

        x = self._encode_perch(waves)

        # (B, C, F, T) with freq at dim 2 to match classic mel layout for pooling
        x = x.permute(0, 1, 3, 2)

        if self.channel_smoothing == "gemfreq":
            x = self.gem_freq_pool(x)
        elif self.channel_smoothing == "max_plus_avg":
            x = torch.mean(x, dim=2)
            x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
            x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
            x = x1 + x2
        else:
            raise NotImplementedError

        x = F.dropout(x, p=self.config["model"]["attn_block"]["dropout"], training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=self.config["model"]["attn_block"]["dropout"], training=self.training)

        clipwise_output, norm_att, segmentwise_output, cla_logit = self.att_block(x)
        clipwise_logit = torch.sum(norm_att * cla_logit, dim=2)
        segmentwise_logit = cla_logit.transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        frames_num = x.size(2)
        interpolate_ratio = frames_num // segmentwise_output.size(1)

        framewise_output = interpolate(segmentwise_output, interpolate_ratio)
        framewise_output = pad_framewise_output(framewise_output, frames_num)

        framewise_logit = interpolate(segmentwise_logit, interpolate_ratio)
        framewise_logit = pad_framewise_output(framewise_logit, frames_num)

        pooling = self.segwise_pooling
        if pooling == "max":
            agg_logit = segmentwise_logit.max(dim=1)[0]
        elif pooling == "mean":
            agg_logit = segmentwise_logit.mean(dim=1)
        elif pooling == "lse":
            r = 10.0
            agg_logit = torch.logsumexp(r * segmentwise_logit, dim=1) / r
        elif pooling == "gem":
            p = 3.0
            z = segmentwise_logit.transpose(1, 2)
            agg_logit = F.avg_pool1d(z.pow(p), kernel_size=z.size(2)).pow(1.0 / p).view(z.size(0), -1)
        elif pooling == "noisy_or":
            probs = torch.sigmoid(segmentwise_logit)
            eps = 1e-7
            bag_probs = 1 - torch.prod(1 - probs + eps, dim=1)
            agg_logit = torch.logit(bag_probs, eps=eps)
        elif pooling == "topk":
            k = 3
            topk_vals, _ = segmentwise_logit.topk(k, dim=1)
            agg_logit = topk_vals.mean(dim=1)
        else:
            raise NotImplementedError

        distill_loss = None
        if y is not None:
            if (
                (
                    bool(self.config["model"].get("multicontext", False))
                    or bool(self.config["model"].get("chunked_multicontext", False))
                )
                and y_bins is not None
                and y_bins.ndim == 3
                and y_bins.size(1) > 0
            ):
                agg_loss = self.criterion(agg_logit, y)
                per_bin_logits = self.segmentwise_bin_max_logits(segmentwise_logit, y_bins.size(1))
                if per_bin_logits is not None:
                    k_eff = min(per_bin_logits.size(1), y_bins.size(1))
                    if k_eff > 0:
                        per_bin_loss = []
                        for i in range(k_eff):
                            per_bin_loss.append(self.criterion(per_bin_logits[:, i, :], y_bins[:, i, :]))
                        agg_loss = torch.stack(per_bin_loss).mean()
                main_loss = agg_loss
            else:
                main_loss = 0.5 * self.criterion(clipwise_logit, y) + 0.5 * self.criterion(agg_logit, y)
            loss = main_loss
        else:
            loss = None

        return {
            "framewise_output": framewise_output,
            "framewise_logit": framewise_logit,
            "segmentwise_output": segmentwise_output,
            "segmentwise_logit": segmentwise_logit,
            "clipwise_output": clipwise_output,
            "clipwise_logit": clipwise_logit,
            "agg_output": torch.sigmoid(agg_logit),
            "agg_logit": agg_logit,
            "logit": clipwise_logit,
            "distill_loss": distill_loss,
            "loss": loss,
        }
