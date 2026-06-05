import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15, sparsemax


def attn_block_kw_from_cfg(attn_cfg):
    return {
        k: attn_cfg[k]
        for k in {
            "activation",
            "att_activation",
            "norm",
            "eps",
            "use_se",
            "use_gru_before",
            "use_gru_after",
            "use_complex_convs",
            "se_reduction",
            "gru_before_hidden",
            "gru_after_hidden",
            "gru_layers",
        }
        if k in attn_cfg
    }


def interpolate(x, ratio):
    batch_size, time_steps, classes_num = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    return F.interpolate(
        framewise_output.unsqueeze(1),
        size=(frames_num, framewise_output.size(2)),
        align_corners=True,
        mode="bilinear",
    ).squeeze(1)


def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (b, c, t)
        scale = self.fc(x).unsqueeze(-1)  # (b, c, 1)
        return x * scale


class AttBlockV2(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "sigmoid",
        att_activation: str = "tanh",
        norm: str = "softmax",
        eps: float = 1e-7,
        use_se: bool = False,
        use_gru_before: bool = False,
        use_gru_after: bool = False,
        use_complex_convs: bool = False,
        se_reduction: int = 16,
        gru_before_hidden: int | None = None,
        gru_after_hidden: int | None = None,
        gru_layers: int = 1,
    ):
        super().__init__()

        self.activation = activation
        self.att_activation = att_activation
        self.norm = norm
        self.eps = eps
        self.use_se = use_se
        self.use_gru_before = use_gru_before
        self.use_gru_after = use_gru_after
        self.use_complex_convs = use_complex_convs

        if self.use_se:
            self.se = SEBlock(in_features, reduction=se_reduction)

        if self.use_gru_before:
            gru_before_hidden = gru_before_hidden if gru_before_hidden is not None else in_features
            self.gru_before = nn.GRU(
                in_features, gru_before_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True
            )
            if 2 * gru_before_hidden != in_features:
                self.gru_before_proj = nn.Linear(2 * gru_before_hidden, in_features)
            else:
                self.gru_before_proj = nn.Identity()

        if self.use_complex_convs:
            self.att = nn.Sequential(
                nn.Conv1d(in_features, in_features, 1, bias=False),
                nn.BatchNorm1d(in_features),
                nn.ReLU(),
                nn.Conv1d(in_features, out_features, 1, bias=True),
            )
            self.cla = nn.Sequential(
                nn.Conv1d(in_features, in_features, 1, bias=False),
                nn.BatchNorm1d(in_features),
                nn.ReLU(),
                nn.Conv1d(in_features, out_features, 1, bias=True),
            )
        else:
            self.att = nn.Conv1d(
                in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=True
            )
            self.cla = nn.Conv1d(
                in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=True
            )

        if self.use_gru_after:
            gru_after_hidden = gru_after_hidden if gru_after_hidden is not None else out_features
            self.gru_att = nn.GRU(
                out_features, gru_after_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True
            )
            self.gru_cla = nn.GRU(
                out_features, gru_after_hidden, num_layers=gru_layers, batch_first=True, bidirectional=True
            )
            if 2 * gru_after_hidden != out_features:
                self.gru_att_proj = nn.Linear(2 * gru_after_hidden, out_features)
                self.gru_cla_proj = nn.Linear(2 * gru_after_hidden, out_features)
            else:
                self.gru_att_proj = nn.Identity()
                self.gru_cla_proj = nn.Identity()

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        if self.use_complex_convs:
            for module in self.att.modules():
                if isinstance(module, nn.Conv1d):
                    init_layer(module)
                elif isinstance(module, nn.BatchNorm1d):
                    init_bn(module)
            for module in self.cla.modules():
                if isinstance(module, nn.Conv1d):
                    init_layer(module)
                elif isinstance(module, nn.BatchNorm1d):
                    init_bn(module)
        else:
            init_layer(self.att)
            init_layer(self.cla)

        if self.use_gru_before:
            for name, param in self.gru_before.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    param.data.fill_(0)
            if hasattr(self.gru_before_proj, "weight"):
                init_layer(self.gru_before_proj)

        if self.use_gru_after:
            for gru in [self.gru_att, self.gru_cla]:
                for name, param in gru.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        param.data.fill_(0)
            if hasattr(self.gru_att_proj, "weight"):
                init_layer(self.gru_att_proj)
            if hasattr(self.gru_cla_proj, "weight"):
                init_layer(self.gru_cla_proj)

        init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_ dtime)
        # norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        # norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)

        if self.use_se:
            x = self.se(x)

        if self.use_gru_before:
            x = x.transpose(1, 2)
            x, _ = self.gru_before(x)
            x = self.gru_before_proj(x)
            x = x.transpose(1, 2)

        att_raw = self.att(x)
        cla_raw = self.cla(x)

        if self.use_gru_after:
            att_t = att_raw.transpose(1, 2)
            att_t, _ = self.gru_att(att_t)
            att_t = self.gru_att_proj(att_t)
            att_raw = att_t.transpose(1, 2)
            cla_t = cla_raw.transpose(1, 2)
            cla_t, _ = self.gru_cla(cla_t)
            cla_t = self.gru_cla_proj(cla_t)
            cla_raw = cla_t.transpose(1, 2)

        if self.att_activation == "tanh":
            att_activated = torch.tanh(att_raw)
        elif self.att_activation == "relu":
            att_activated = F.relu(att_raw)
        elif self.att_activation == "gelu":
            att_activated = F.gelu(att_raw)
        else:
            att_activated = torch.clamp(att_raw, -10, 10)

        if self.norm == "softmax":
            norm_att = F.softmax(att_activated, dim=-1)
        elif self.norm == "sparsemax":
            norm_att = sparsemax(att_activated, dim=-1)
        elif self.norm == "entmax15":
            norm_att = entmax15(att_activated, dim=-1)
        elif self.norm == "sigmoid_norm":
            att_sigm = torch.sigmoid(att_activated)
            norm_att = att_sigm / (att_sigm.sum(dim=-1, keepdim=True) + self.eps)
        else:
            raise NotImplementedError

        cla = self.nonlinear_transform(cla_raw)
        if self.activation == "sigmoid":
            # pool logits then sigmoid — matches bcewithlogits without logit(clip)
            clipwise_logit = torch.sum(norm_att * cla_raw, dim=2)
            out = torch.sigmoid(clipwise_logit)
        else:
            out = torch.sum(norm_att * cla, dim=2)
        return out, norm_att, cla, cla_raw

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class GeMFreqPool(nn.Module):
    def __init__(self, p_init=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = eps

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=self.eps).pow(p)
        x = x.mean(dim=2)
        return x.pow(1.0 / p)
