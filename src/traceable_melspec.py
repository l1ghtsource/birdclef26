import torch
import torch.nn as nn
from torchaudio.transforms import MelScale

try:
    from nnAudio.features.stft import STFT as nnAudioSTFT
except ImportError:  # pragma: no cover
    nnAudioSTFT = None  # type: ignore[misc, assignment]


class TraceableMelspec(nn.Module):
    def __init__(
        self,
        win_length: int | None = None,
        hop_length: int | None = None,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: float | None = None,
        n_fft: int = 400,
        norm: str | None = None,
        mel_scale: str = "htk",
        trainable: bool = False,
        quantizable: bool = False,
        *,
        stft_verbose: bool = False,
    ):
        super().__init__()
        if quantizable:
            raise NotImplementedError("quantizable TraceableMelspec is not ported to birds_hand")
        if nnAudioSTFT is None:
            raise ImportError("TraceableMelspec requires nnAudio. Install with: pip install nnaudio>=0.3.3,<0.4")
        self.spectrogram = nnAudioSTFT(
            n_fft=n_fft,
            win_length=win_length,
            freq_bins=None,
            hop_length=hop_length,
            window="hann",
            freq_scale="no",
            center=center,
            pad_mode=pad_mode,
            iSTFT=False,
            sr=sample_rate,
            trainable=trainable,
            output_format="Complex",
            verbose=stft_verbose,
        )
        self.normalized = normalized
        self.power = power
        self.register_buffer(
            "window",
            torch.hann_window(win_length if win_length is not None else n_fft),
        )
        self.trainable = trainable
        self.mel_scale = MelScale(n_mels, sample_rate, f_min, f_max, n_fft // 2 + 1, norm, mel_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec_f = self.spectrogram(x)
        if self.normalized:
            spec_f = spec_f / self.window.pow(2.0).sum().sqrt()
        if self.power is not None:
            eps = 1e-8 if self.trainable else 0.0
            spec_f = torch.sqrt(spec_f[:, :, :, 0].pow(2) + spec_f[:, :, :, 1].pow(2) + eps)
            if self.power != 1.0:
                spec_f = spec_f.pow(self.power)
        return self.mel_scale(spec_f)
