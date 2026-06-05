import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio

VAD_SAMPLE_RATE = 16_000
MIN_VAD_AUDIO_DURATION_S = 1.0
MIN_VOICE_SEGMENT_DURATION_S = 1.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_silero_vad_repo(project_root: Path | None = None) -> Path:
    r = repo_root() if project_root is None else project_root
    return (r / "models" / "silero-vad").resolve()


def resolve_silero_vad_repo(repo_dir: Path | str | None) -> Path:
    if repo_dir is None:
        return default_silero_vad_repo()
    p = Path(repo_dir).resolve()
    if (p / "src" / "silero_vad").is_dir():
        return p
    nested = p / "models" / "silero-vad"
    if nested.is_dir():
        return nested.resolve()
    return p


def default_silero_jit_path(silero_repo: Path) -> Path:
    return (silero_repo / "src" / "silero_vad" / "data" / "silero_vad.jit").resolve()


def _ensure_silero_import_path(repo_root_dir: Path) -> None:
    src = str((repo_root_dir / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)


def load_silero_vad(
    repo_dir: Path | str | None = None,
    *,
    jit_path: Path | str | None = None,
    use_gpu: bool = False,
) -> tuple[Any, Callable[..., list]]:
    torch.set_num_threads(1)
    repo = resolve_silero_vad_repo(repo_dir)
    jp = Path(jit_path) if jit_path is not None else default_silero_jit_path(repo)
    if not jp.is_file():
        raise FileNotFoundError(f"Silero VAD JIT not found: {jp}")

    _ensure_silero_import_path(repo)
    from silero_vad.utils_vad import get_speech_timestamps, init_jit_model

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = init_jit_model(str(jp), device=device)
    model = model.to(device)
    return model, get_speech_timestamps


def _mono_float_tensor_from_file(path: Path) -> tuple[torch.Tensor, int]:
    w, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if w.ndim == 2 and w.shape[1] > 1:
        w = np.mean(w, axis=1)
    else:
        w = w.reshape(-1)
    t = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
    if int(sr) != VAD_SAMPLE_RATE:
        t = t.unsqueeze(0)
        t = torchaudio.functional.resample(t, int(sr), VAD_SAMPLE_RATE)
        t = t.squeeze(0)
    return t, VAD_SAMPLE_RATE


def detect_voice_timestamps(
    path: Path | str,
    model: Any,
    get_speech_timestamps: Callable[..., list],
    *,
    threshold: float = 0.9,
    min_speech_duration_ms: int | None = None,
) -> list[tuple[float, float]]:
    path = Path(path)
    info = sf.info(str(path))
    dur_native = float(info.frames) / float(info.samplerate)
    if dur_native < MIN_VAD_AUDIO_DURATION_S:
        return []

    wav, sr = _mono_float_tensor_from_file(path)
    if min_speech_duration_ms is None:
        min_speech_duration_ms = int(MIN_VOICE_SEGMENT_DURATION_S * 1000)

    device = next(model.parameters()).device
    wav = wav.to(device)

    speeches = get_speech_timestamps(
        wav,
        model,
        threshold=threshold,
        sampling_rate=sr,
        min_speech_duration_ms=min_speech_duration_ms,
        return_seconds=True,
        time_resolution=3,
    )
    out: list[tuple[float, float]] = []
    for seg in speeches:
        if not isinstance(seg, dict):
            continue
        try:
            t0 = float(seg["start"])
            t1 = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if t1 > t0:
            out.append((t0, t1))
    return out
