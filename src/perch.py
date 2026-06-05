import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_perchv2_pytorch_module():
    path = _repo_root() / "models" / "pytorch_perchv2" / "perchv2_pytorch.py"
    if not path.is_file():
        raise FileNotFoundError(f"perchv2_pytorch not found: {path}")
    spec = importlib.util.spec_from_file_location("birds_hand_perchv2_pytorch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_perchv2_pytorch():
    return _load_perchv2_pytorch_module()


def prepare_wave_fixed_5s(w: torch.Tensor, n_samples: int = 160000) -> torch.Tensor:
    if w.dim() == 3 and w.size(1) == 1:
        w = w.squeeze(1)
    if w.dim() != 2:
        raise ValueError(f"expected (B, T) or (B, 1, T), got {tuple(w.shape)}")
    t = w.size(1)
    if t > n_samples:
        start = (t - n_samples) // 2
        w = w[:, start : start + n_samples]
    elif t < n_samples:
        w = F.pad(w, (0, n_samples - t))
    return w


class OnnxFeatureExtractor(nn.Module):
    def __init__(self, onnx_path: str | Path):
        super().__init__()
        import onnxruntime as ort

        onnx_path = str(Path(onnx_path))
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Perch ONNX expects (B, T), got {tuple(x.shape)}")

        x = x.contiguous()

        if x.is_cuda:
            device_id = x.device.index if x.device.index is not None else 0
            batch_size = x.shape[0]
            embedding = torch.empty((batch_size, 1536), device=x.device, dtype=torch.float32)
            spatial_embedding = torch.empty((batch_size, 16, 4, 1536), device=x.device, dtype=torch.float32)

            binding = self.session.io_binding()
            binding.bind_input(
                name="inputs",
                device_type="cuda",
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(x.shape),
                buffer_ptr=x.data_ptr(),
            )
            binding.bind_output(
                name="embedding",
                device_type="cuda",
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(embedding.shape),
                buffer_ptr=embedding.data_ptr(),
            )
            binding.bind_output(
                name="spatial_embedding",
                device_type="cuda",
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(spatial_embedding.shape),
                buffer_ptr=spatial_embedding.data_ptr(),
            )
            self.session.run_with_iobinding(binding)
            return embedding

        out = self.session.run(None, {"inputs": x.cpu().numpy().astype(np.float32, copy=False)})

        for arr in out:
            if arr.ndim == 2 and arr.shape[1] == 1536:
                return torch.from_numpy(arr).to(x.device, dtype=torch.float32)
        raise RuntimeError("Perch ONNX did not return 1536-d embedding output.")


class TorchPerchFeatureExtractor(nn.Module):
    def __init__(self, frontend_weights: str | Path, backbone_weights: str | Path):
        super().__init__()
        mod = _load_perchv2_pytorch_module()
        self.frontend = mod.Perch2Frontend()
        self.backbone = mod.Perch2EfficientB3()
        fw = torch.load(Path(frontend_weights), map_location="cpu", weights_only=True)
        bw = torch.load(Path(backbone_weights), map_location="cpu", weights_only=True)
        self.frontend.load_state_dict(fw, strict=True)
        self.backbone.load_state_dict(bw, strict=True)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        if x.dtype != torch.float32:
            x = x.float()
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Torch Perch expects (B, T), got {tuple(x.shape)}")
        x = x.contiguous()
        spec = self.frontend(x)
        return self.backbone(spec)


class HybridPerchExtractor(nn.Module):
    def __init__(
        self,
        onnx_path: str | Path,
        frontend_weights: str | Path,
        backbone_weights: str | Path,
        *,
        n_fixed_samples: int = 160000,
        lazy_torch: bool = True,
    ):
        super().__init__()
        self.onnx_ext = OnnxFeatureExtractor(onnx_path)
        self._frontend_w = Path(frontend_weights)
        self._backbone_w = Path(backbone_weights)
        self.n_fixed_samples = int(n_fixed_samples)
        self.torch_perch: TorchPerchFeatureExtractor | None = None
        if not bool(lazy_torch):
            tp = TorchPerchFeatureExtractor(self._frontend_w, self._backbone_w)
            self.torch_perch = tp
            self.add_module("torch_perch", tp)

    def _ensure_torch(self, device: torch.device) -> None:
        if self.torch_perch is None:
            tp = TorchPerchFeatureExtractor(self._frontend_w, self._backbone_w)
            self.torch_perch = tp
            self.add_module("torch_perch", tp)
        self.torch_perch.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Hybrid Perch expects (B, T) or (B, 1, T), got {tuple(x.shape)}")
        t = x.size(1)
        if t <= self.n_fixed_samples:
            x5 = prepare_wave_fixed_5s(x, self.n_fixed_samples)
            return self.onnx_ext(x5)
        self._ensure_torch(x.device)
        return self.torch_perch(x)
