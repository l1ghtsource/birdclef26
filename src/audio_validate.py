import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

_FFMPEG_DECODE_EXTS = frozenset({".mp3", ".m4a", ".aac", ".mp4", ".ogg", ".oga", ".opus"})

VALIDATE_AUDIO_DEFAULT_WORKERS = 32


def resolve_ffmpeg_executable() -> str | None:
    for key in ("FFMPEG_BINARY", "AUDIO_VALIDATE_FFMPEG"):
        raw = os.environ.get(key, "").strip()
        if raw and Path(raw).is_file():
            return raw
    w = shutil.which("ffmpeg")
    if w:
        return w
    repo_root = Path(__file__).resolve().parent.parent
    parent = repo_root.parent
    static = parent / "ffmpeg-7.0.2-amd64-static" / "ffmpeg"
    if static.is_file():
        return str(static)
    for cand in sorted(parent.glob("ffmpeg-*-static/ffmpeg")):
        if cand.is_file():
            return str(cand)
    return None


def _ffmpeg_sniff_decode_ok(path: Path, *, max_sec: float = 2.0, timeout_s: float = 90.0) -> tuple[bool, str]:
    exe = resolve_ffmpeg_executable()
    if exe is None:
        return True, ""
    try:
        p = subprocess.run(
            [
                exe,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-xerror",
                "-i",
                str(path),
                "-t",
                str(max_sec),
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, "ffmpeg_timeout"
    except OSError as e:
        return False, f"ffmpeg_os:{e}"
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        return False, (msg or "ffmpeg_nonzero_rc")[:800]
    return True, ""


def _read_chunk_soundfile(path: Path, *, max_frames: int) -> tuple[np.ndarray, int] | None:
    try:
        info = sf.info(str(path))
    except Exception:
        return None
    sr = int(info.samplerate)
    total = int(info.frames)
    if total <= 0:
        return np.zeros((0, 1), dtype=np.float32), sr
    n = min(total, max_frames)
    try:
        data, _sr = sf.read(str(path), dtype="float32", start=0, frames=n, always_2d=True)
    except Exception:
        return None
    return np.ascontiguousarray(data), sr


def _read_chunk_torchaudio(path: Path, *, max_frames: int) -> tuple[torch.Tensor, int] | None:
    try:
        wav, sr = torchaudio.load(str(path), frame_offset=0, num_frames=max_frames, normalize=True)
    except Exception:
        return None
    if wav.numel() == 0:
        return None
    return wav, int(sr)


def validate_audio_file(
    path: Path,
    *,
    target_sr: int = 32000,
    max_read_sec: float = 6.0,
    use_ffmpeg_sniff: bool = True,
) -> tuple[bool, str]:
    path = Path(path)
    if not path.is_file():
        return False, "not_a_file"

    ext = path.suffix.lower()
    if use_ffmpeg_sniff and ext in _FFMPEG_DECODE_EXTS:
        ok_ff, why = _ffmpeg_sniff_decode_ok(path)
        if not ok_ff:
            return False, f"ffmpeg:{why}"

    max_frames_sf = max(1, int(max_read_sec * 48000))
    chunk = _read_chunk_soundfile(path, max_frames=max_frames_sf)
    wav: torch.Tensor | None = None
    sr: int | None = None
    if chunk is not None:
        data, sr = chunk
        if data.size == 0:
            info = sf.info(str(path))
            if int(info.frames) > 0:
                return False, "soundfile_empty_frames"
        if not np.isfinite(data).all():
            return False, "soundfile_nonfinite"
        wav = torch.from_numpy(np.ascontiguousarray(data.T))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
    else:
        ta = _read_chunk_torchaudio(path, max_frames=max_frames_sf)
        if ta is None:
            return False, "decode_failed_soundfile_and_torchaudio"
        wav, sr = ta
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if not torch.isfinite(wav).all():
            return False, "torchaudio_nonfinite"

    assert wav is not None and sr is not None
    if wav.shape[1] == 0:
        return False, "zero_length_waveform"
    if sr != target_sr:
        try:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        except Exception as e:
            return False, f"resample:{e}"
    if not torch.isfinite(wav).all():
        return False, "nonfinite_after_resample"
    return True, ""
