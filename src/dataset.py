import contextlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader

from src.melspec import AudioToSpec
from src.samplers import get_pseudo_balanced_batch_sampler, get_sampler, get_source_balanced_batch_sampler
from src.taxonomy_merge import build_merged_label_to_idx
from src.utils import filter_taxonomy_labels, norm_class_label, parse_list_cell, time_hms_to_seconds
from src.wave_aug import apply_wave_awgn_snr, apply_wave_random_gain_db

_SILENCE_DECODER_STDERR_EXTS = frozenset({".mp3", ".m4a", ".aac", ".mp4"})


@contextlib.contextmanager
def _maybe_silence_decoder_stderr(path: Path):
    ext = path.suffix.lower()
    if ext not in _SILENCE_DECODER_STDERR_EXTS:
        yield
        return
    if str(os.environ.get("BIRDS_HAND_SILENCE_DECODER_STDERR", "")).strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        yield
        return
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        yield
        return
    saved = os.dup(fd)
    dn = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(dn, fd)
        yield
    finally:
        os.dup2(saved, fd)
        os.close(saved)
        os.close(dn)


def _ds_cfg_get(ds_cfg, key, default=None):
    if isinstance(ds_cfg, dict):
        return ds_cfg.get(key, default)
    return getattr(ds_cfg, key, default)


def load_audio_blocklist_rel(data_root: Path, rel_txt: str | None) -> frozenset[str]:
    if not rel_txt:
        return frozenset()
    p = data_root / str(rel_txt)
    if not p.is_file():
        return frozenset()
    out: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(Path(line).as_posix())
    return frozenset(out)


def load_merged_audio_blocklists(data_root: Path, ds_cfg: dict) -> frozenset[str]:
    rels: list[str] = []
    one = _ds_cfg_get(ds_cfg, "audio_blocklist_rel")
    if one:
        rels.append(str(one))
    extra = _ds_cfg_get(ds_cfg, "audio_blocklist_rels")
    if extra:
        if isinstance(extra, (str, bytes)):
            rels.append(str(extra))
        else:
            rels.extend(str(x) for x in extra if x)
    merged: set[str] = set()
    for rel in rels:
        merged |= load_audio_blocklist_rel(data_root, rel)
    return frozenset(merged)


def item_rel_under_data_root(item: "DatasetItem", data_root: Path) -> str | None:
    try:
        return item.path.resolve().relative_to(data_root.resolve()).as_posix()
    except ValueError:
        return None


def train_audio_csv_specs(ds_cfg):
    out = []
    if _ds_cfg_get(ds_cfg, "use_train_audio", False):
        out.append(
            (
                _ds_cfg_get(ds_cfg, "train_csv"),
                _ds_cfg_get(ds_cfg, "train_audio_subdir", "") or "",
            )
        )
    for block in _ds_cfg_get(ds_cfg, "additional_train_csv") or []:
        if isinstance(block, dict):
            csv_rel = block["train_csv"]
            sub = block.get("train_audio_subdir", "") or ""
        else:
            csv_rel = block.train_csv
            sub = getattr(block, "train_audio_subdir", "") or ""
        out.append((csv_rel, sub))
    return out


def peak_normalize_waveform(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = wav.abs().max()
    if m < eps:
        return wav
    return wav / m


def reshape_waveform(wav_1ch, width):
    T = int(wav_1ch.shape[1])
    if T < 2:
        raise ValueError("waveform too short")
    if T % 2 != 0:
        wav_1ch = wav_1ch[:, :-1]
        T -= 1
    half = T // 2
    usable = (half // width) * width
    if usable == 0:
        raise ValueError(f"chunk too short for {width=}")
    if usable < half:
        wav_1ch = wav_1ch[:, : usable * 2]
    m = torch.amax(torch.abs(wav_1ch))
    if m > 1:
        wav_1ch = wav_1ch / m
    x = wav_1ch.view(1, -1, 2)
    x = x.transpose(1, 2)
    x = x.view(1, 2, -1, width).squeeze(0)
    x3 = torch.cat([x[0:1], x[1:2], (x[0:1] + x[1:2]) * 0.5], dim=0)
    return x3


def load_audio_soundfile(path, frame_offset=0, num_frames=-1):
    path = Path(path)
    with _maybe_silence_decoder_stderr(path):
        info = sf.info(str(path))
        sr = int(info.samplerate)
        total = int(info.frames)
        ch = int(info.channels)
        frame_offset = max(0, frame_offset)
        if frame_offset >= total:
            return torch.zeros(ch, 0, dtype=torch.float32), sr
        if num_frames < 0:
            to_read = total - frame_offset
        else:
            to_read = min(num_frames, total - frame_offset)
        data, _sr_read = sf.read(str(path), dtype="float32", start=frame_offset, frames=to_read, always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.T))
    return wav, sr


class CropMode(str, Enum):
    RANDOM = "random"
    RMS = "rms"
    START_RANDOM = "start_random"
    END_RANDOM = "end_random"


class PaddingMode(str, Enum):
    LEFT = "left"
    CENTERED = "centered"
    RANDOM_PLACE = "random_place"
    REPEATED = "repeated"


@dataclass(frozen=True)
class DatasetItem:
    source: str
    path: Path
    primary_labels: tuple[str, ...]
    secondary_labels: tuple[str, ...]
    t0_sec: float | None = None
    bin_primary_labels: tuple[tuple[str, ...], ...] | None = None
    soft_target: tuple[float, ...] | None = None
    bin_soft_targets: tuple[tuple[float, ...], ...] | None = None
    is_pseudo: bool = False


def expand_train_indices_for_upsampling(tr_idx, fold_items, upsampling_n, seed):
    if upsampling_n <= 0:
        return tr_idx
    cnt: Counter[str] = Counter()
    for it in fold_items:
        for lab in it.primary_labels:
            cnt[lab] += 1
    rng = np.random.default_rng(int(seed))
    out = list(tr_idx)
    for c, n in cnt.items():
        if n >= upsampling_n:
            continue
        pool = [tr_idx[i] for i, it in enumerate(fold_items) if c in it.primary_labels]
        if not pool:
            continue
        need = upsampling_n - n
        extra = rng.choice(np.asarray(pool, dtype=np.int64), size=need, replace=True)
        out.extend(int(x) for x in extra.tolist())
    rng.shuffle(out)
    return out


def pad_waveform_to_length(waveform, length, mode, rng, train):
    _, n = waveform.shape
    if n >= length:
        return waveform
    pad = length - n
    if mode == PaddingMode.LEFT:
        return torch.nn.functional.pad(waveform, (0, pad))
    if mode == PaddingMode.CENTERED:
        pl = pad // 2
        pr = pad - pl
        return torch.nn.functional.pad(waveform, (pl, pr))
    if mode == PaddingMode.RANDOM_PLACE:
        if train:
            pl = int(torch.randint(0, pad + 1, (1,), generator=rng).item())
        else:
            pl = pad // 2
        pr = pad - pl
        return torch.nn.functional.pad(waveform, (pl, pr))
    if mode == PaddingMode.REPEATED:
        if n <= 0:
            return torch.zeros(waveform.shape[0], length, dtype=waveform.dtype, device=waveform.device)
        reps = (length + n - 1) // n
        tiled = waveform.repeat(1, int(reps))
        return tiled[:, :length]
    raise NotImplementedError


def _pick_regular_crop_start(waveform, n: int, chunk: int, mode: CropMode, rng) -> int:
    max_start = n - chunk
    if mode == CropMode.RANDOM:
        return int(torch.randint(0, max_start + 1, (1,), generator=rng).item())
    if mode == CropMode.START_RANDOM:
        hi = min(2 * chunk, max_start)
        return int(torch.randint(0, hi + 1, (1,), generator=rng).item())
    if mode == CropMode.END_RANDOM:
        lo = max(0, max_start - 2 * chunk)
        return int(torch.randint(lo, max_start + 1, (1,), generator=rng).item())
    if mode == CropMode.RMS:
        mono = waveform[0] if waveform.shape[0] > 0 else waveform.mean(dim=0)
        step = max(1, max_start // 200) if max_start > 200 else 1
        best_start = 0
        best_score = -1.0
        s = 0
        while s <= max_start:
            win = mono[s : s + chunk]
            score = torch.sqrt(torch.mean(win * win) + 1e-12).item()
            if score > best_score:
                best_score = score
                best_start = s
            s += step
        return best_start
    raise NotImplementedError(mode)


def crop_audio(waveform, chunk, mode, rng, padding_mode, train):
    _, n = waveform.shape
    if n == chunk:
        return waveform
    if n < chunk:
        return pad_waveform_to_length(waveform, chunk, padding_mode, rng, train)
    start = _pick_regular_crop_start(waveform, n, chunk, mode, rng)
    return waveform[:, start : start + chunk]


_VOICE_REGION_CSV_STEMS = ("train_audio", "xc", "inat", "tsa", "redownloaded_corrupted")

_DEFAULT_REMOVE_VOICES_SOURCES: frozenset[str] = frozenset(("train", "inat", "xc", "tsa", "redownloaded_corrupted"))
_REMOVE_VOICES_SOURCE_ALIASES: dict[str, str] = {"train_audio": "train"}


def _normalize_remove_voices_sources(raw) -> frozenset[str]:
    if raw is None:
        return _DEFAULT_REMOVE_VOICES_SOURCES
    if isinstance(raw, (str, bytes)):
        seq = (raw,)
    else:
        seq = raw
    out: set[str] = set()
    for s in seq:
        t = str(s).strip()
        if not t:
            continue
        t = _REMOVE_VOICES_SOURCE_ALIASES.get(t, t)
        if t in _DEFAULT_REMOVE_VOICES_SOURCES:
            out.add(t)
    return frozenset(out)


def _voice_remove_bucket(source: str) -> str | None:
    if source == "train_audio":
        return "train_audio"
    if source == "extra_xc":
        return "xc"
    if source == "extra_inat":
        return "inat"
    if source == "extra_tsa":
        return "tsa"
    return None


def _voice_regions_index_stats(regions_by_rel: dict[str, list[tuple[float, float]]]) -> dict[str, float | int | None]:
    files_with_voice = 0
    segments_total = 0
    durs: list[float] = []
    for regs in regions_by_rel.values():
        if regs:
            files_with_voice += 1
        for t0, t1 in regs:
            segments_total += 1
            d = float(t1) - float(t0)
            if d > 0:
                durs.append(d)
    n_rows = len(regions_by_rel)
    if not durs:
        return {
            "indexed_paths": n_rows,
            "files_with_voice": 0,
            "segments_total": 0,
            "seg_dur_s_min": None,
            "seg_dur_s_max": None,
            "seg_dur_s_mean": None,
        }
    arr = np.asarray(durs, dtype=np.float64)
    return {
        "indexed_paths": n_rows,
        "files_with_voice": int(files_with_voice),
        "segments_total": int(segments_total),
        "seg_dur_s_min": float(arr.min()),
        "seg_dur_s_max": float(arr.max()),
        "seg_dur_s_mean": float(arr.mean()),
    }


def _parse_voice_regions_cell(val) -> list[tuple[float, float]]:
    if val is None:
        return []
    if isinstance(val, float) and np.isnan(val):
        return []
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        data = json.loads(s)
    else:
        data = val
    if not isinstance(data, list):
        return []
    out: list[tuple[float, float]] = []
    for pair in data:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        t0, t1 = float(pair[0]), float(pair[1])
        if t1 > t0:
            out.append((t0, t1))
    return out


def _merge_int_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged: list[list[int]] = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(int(a[0]), int(a[1])) for a in merged]


def _bad_sample_intervals_from_voice(
    regions_sec: list[tuple[float, float]], n: int, sample_rate: int
) -> list[tuple[int, int]]:
    bad: list[tuple[int, int]] = []
    for t0, t1 in regions_sec:
        a = int(np.floor(t0 * sample_rate))
        b = int(np.ceil(t1 * sample_rate))
        a = max(0, min(a, n))
        b = max(0, min(b, n))
        if b > a:
            bad.append((a, b))
    return _merge_int_intervals(bad)


def _forbidden_start_intervals(bad_spans: list[tuple[int, int]], n: int, chunk: int) -> list[tuple[int, int]]:
    max_start = n - chunk
    if max_start < 0:
        return []
    forb: list[tuple[int, int]] = []
    for a, b in bad_spans:
        lo = max(0, a - chunk + 1)
        hi = min(max_start, b - 1)
        if lo <= hi:
            forb.append((lo, hi))
    return _merge_int_intervals(forb)


def _complement_intervals(forbidden: list[tuple[int, int]], lo: int, hi: int) -> list[tuple[int, int]]:
    if lo > hi:
        return []
    good: list[tuple[int, int]] = []
    cur = lo
    for a, b in forbidden:
        if a > hi:
            break
        if cur < a:
            good.append((cur, min(a - 1, hi)))
        cur = max(cur, b + 1)
        if cur > hi:
            return [(x, y) for x, y in good if x <= y]
    if cur <= hi:
        good.append((cur, hi))
    return [(x, y) for x, y in good if x <= y]


def _intersect_start_intervals(good: list[tuple[int, int]], lo_b: int, hi_b: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for lo, hi in good:
        a = max(lo, lo_b)
        b = min(hi, hi_b)
        if a <= b:
            out.append((a, b))
    return out


def _random_start_from_good(good: list[tuple[int, int]], rng) -> int:
    lengths = [hi - lo + 1 for lo, hi in good]
    total = sum(lengths)
    if total <= 0:
        return 0
    u = int(torch.randint(0, total, (1,), generator=rng).item())
    acc = 0
    for (lo, _hi), ln in zip(good, lengths, strict=True):
        if u < acc + ln:
            return lo + (u - acc)
        acc += ln
    return good[-1][0]


def _voice_overlap_samples(start: int, chunk: int, bad_merged: list[tuple[int, int]]) -> int:
    end = start + chunk
    tot = 0
    for a, b in bad_merged:
        lo = max(start, a)
        hi = min(end, b)
        if hi > lo:
            tot += hi - lo
    return tot


def _voice_overlap_log_fields(
    start: int, chunk: int, bad_merged: list[tuple[int, int]], sample_rate: int
) -> dict[str, Any]:
    ov = _voice_overlap_samples(start, chunk, bad_merged)
    frac = float(ov) / float(chunk) if chunk else 0.0
    return {
        "voice_overlap_samples": int(ov),
        "voice_overlap_frac": round(frac, 5),
        "voice_overlap_duration_s": round(float(ov) / float(sample_rate), 5) if sample_rate else None,
        "chunk_duration_s": round(float(chunk) / float(sample_rate), 5) if sample_rate else None,
        "crop_start_sample": int(start),
    }


def _best_start_min_voice_overlap(max_start: int, chunk: int, bad_merged: list[tuple[int, int]], rng) -> int:
    if max_start <= 0:
        return 0
    cand: set[int] = {0, max_start}
    for a, b in bad_merged:
        for x in (a - chunk + 1, a, b - chunk, b - chunk + 1, (a + b - chunk) // 2):
            xi = int(np.clip(int(x), 0, max_start))
            cand.add(xi)
    step = max(1, max_start // 256)
    for s in range(0, max_start + 1, step):
        cand.add(int(s))
    for _ in range(32):
        cand.add(int(torch.randint(0, max_start + 1, (1,), generator=rng).item()))
    scored = [(_voice_overlap_samples(s, chunk, bad_merged), s) for s in cand]
    min_o = min(o for o, _ in scored)
    pool = [s for o, s in scored if o == min_o]
    j = int(torch.randint(0, len(pool), (1,), generator=rng).item())
    return int(pool[j])


def crop_audio_avoid_voice_regions(
    waveform,
    chunk,
    regions_sec,
    sample_rate,
    mode,
    rng,
    padding_mode,
    train,
    *,
    voice_fallback_log: Callable[[str, dict[str, Any] | None], None] | None = None,
    voice_fallback_minimize_overlap: bool = True,
):
    _, n = waveform.shape
    if not regions_sec:
        return crop_audio(waveform, chunk, mode, rng, padding_mode, train)
    if n < chunk:
        return pad_waveform_to_length(waveform, chunk, padding_mode, rng, train)
    max_start = n - chunk
    bad = _bad_sample_intervals_from_voice(regions_sec, n, sample_rate)
    forb = _forbidden_start_intervals(bad, n, chunk)
    good = _complement_intervals(forb, 0, max_start)
    if not good:
        if voice_fallback_minimize_overlap and bad:
            start = _best_start_min_voice_overlap(max_start, chunk, bad, rng)
            if voice_fallback_log is not None:
                ex = _voice_overlap_log_fields(start, chunk, bad, sample_rate)
                voice_fallback_log("no_chunk_without_voice_min_overlap", ex)
            return waveform[:, start : start + chunk]
        start = _pick_regular_crop_start(waveform, n, chunk, mode, rng)
        if voice_fallback_log is not None and bad:
            ex = _voice_overlap_log_fields(start, chunk, bad, sample_rate)
            voice_fallback_log("no_chunk_without_voice", ex)
        elif voice_fallback_log is not None:
            voice_fallback_log("no_chunk_without_voice", None)
        return waveform[:, start : start + chunk]

    if mode == CropMode.START_RANDOM:
        hi_r = min(2 * chunk, max_start)
        good = _intersect_start_intervals(good, 0, hi_r)
    elif mode == CropMode.END_RANDOM:
        lo_r = max(0, max_start - 2 * chunk)
        good = _intersect_start_intervals(good, lo_r, max_start)

    if not good:
        if voice_fallback_minimize_overlap and bad:
            start = _best_start_min_voice_overlap(max_start, chunk, bad, rng)
            if voice_fallback_log is not None:
                ex = _voice_overlap_log_fields(start, chunk, bad, sample_rate)
                voice_fallback_log("no_valid_start_in_crop_window_min_overlap", ex)
            return waveform[:, start : start + chunk]
        start = _pick_regular_crop_start(waveform, n, chunk, mode, rng)
        if voice_fallback_log is not None and bad:
            ex = _voice_overlap_log_fields(start, chunk, bad, sample_rate)
            voice_fallback_log("no_valid_start_in_crop_window", ex)
        elif voice_fallback_log is not None:
            voice_fallback_log("no_valid_start_in_crop_window", None)
        return waveform[:, start : start + chunk]

    if mode in (CropMode.RANDOM, CropMode.START_RANDOM, CropMode.END_RANDOM):
        start = _random_start_from_good(good, rng)
        return waveform[:, start : start + chunk]

    if mode == CropMode.RMS:
        mono = waveform[0] if waveform.shape[0] > 0 else waveform.mean(dim=0)
        best_start = good[0][0]
        best_score = -1.0
        span = sum(hi - lo + 1 for lo, hi in good)
        step = max(1, span // 400) if span > 400 else 1
        for lo, hi in good:
            s = lo
            while s <= hi:
                win = mono[s : s + chunk]
                score = torch.sqrt(torch.mean(win * win) + 1e-12).item()
                if score > best_score:
                    best_score = score
                    best_start = s
                s += step
        return waveform[:, best_start : best_start + chunk]

    raise NotImplementedError(mode)


def collect_noises(data_root, noise_dirs, *, filter_invalid_audio: bool = False):
    paths = []
    for raw in noise_dirs:
        d = Path(raw)
        root = d if d.is_absolute() else (data_root / d)
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3", ".opus"):
                paths.append(p)
    paths = sorted(set(paths))
    if not filter_invalid_audio or not paths:
        return paths
    from src.audio_validate import validate_audio_file

    kept: list[Path] = []
    for p in paths:
        ok, _ = validate_audio_file(p)
        if ok:
            kept.append(p)
    return kept


def filter_noise_paths_min_duration(paths: list[Path], min_duration_s: float) -> tuple[list[Path], int]:
    if min_duration_s <= 0.0:
        return paths, 0
    kept: list[Path] = []
    dropped = 0
    for p in paths:
        try:
            info = sf.info(str(p))
            dur = float(info.frames) / float(info.samplerate)
            if dur >= float(min_duration_s):
                kept.append(p)
            else:
                dropped += 1
        except Exception:
            dropped += 1
    return kept, dropped


def prepare_noise_tensor(noise_1ch, target_len, rng):
    _, t = noise_1ch.shape
    if t == 0:
        return torch.zeros(1, target_len, dtype=noise_1ch.dtype)
    if t >= target_len:
        start = int(torch.randint(0, t - target_len + 1, (1,), generator=rng).item())
        return noise_1ch[:, start : start + target_len].contiguous()
    out = noise_1ch
    while out.shape[1] < target_len:
        out = torch.cat([out, noise_1ch], dim=1)
    return out[:, :target_len].contiguous()


def build_soundscape_items(data_root, csv_path, subdir, label_to_idx, chunk_duration_s, label_bin_s):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["filename", "start", "end", "primary_label"], keep="first").reset_index(drop=True)
    base = data_root / subdir
    n_bins = round(chunk_duration_s / label_bin_s)
    bin_to_labels = {}
    for _, row in df.iterrows():
        fn = str(row["filename"])
        t0 = float(time_hms_to_seconds(row["start"]))
        key = (fn, round(t0, 6))
        labs = [norm_class_label(x) for x in str(row["primary_label"]).split(";") if str(x).strip()]
        labs = filter_taxonomy_labels(labs, label_to_idx)
        if key not in bin_to_labels:
            bin_to_labels[key] = set()
        bin_to_labels[key].update(labs)
    files = sorted({str(row["filename"]) for _, row in df.iterrows()})
    items = []
    for fn in files:
        path = base / fn
        if not path.is_file():
            continue
        info = sf.info(str(path))
        duration = float(info.frames) / float(info.samplerate)
        t_win = 0.0
        while t_win + chunk_duration_s <= duration + 1e-6:
            union = set()
            per_bin_labels = []
            for k in range(n_bins):
                seg_t = round(t_win + k * label_bin_s, 6)
                labs_k = bin_to_labels.get((fn, seg_t), set())
                union |= labs_k
                per_bin_labels.append(tuple(sorted(labs_k)))
            primary = tuple(sorted(union))
            items.append(
                DatasetItem(
                    source="train_soundscapes",
                    path=path,
                    primary_labels=primary,
                    secondary_labels=(),
                    t0_sec=t_win,
                    bin_primary_labels=tuple(per_bin_labels),
                )
            )
            t_win += label_bin_s
    return items


def parse_row_id_window(row_id: str):
    rid = str(row_id)
    stem, end_s_raw = rid.rsplit("_", 1)
    end_s = int(end_s_raw)
    start_s = end_s - 5
    if start_s < 0:
        return None
    fn = f"{stem}.ogg"
    return fn, start_s


def build_soundscape_pseudo_items(
    data_root,
    pseudo_csv_path,
    labeled_csv_path,
    subdir,
    label_to_idx,
    chunk_duration_s,
    label_bin_s,
    *,
    pl_filter,
    pl_filter_thr,
    pl_zero_unconf,
    pl_zero_unconf_thr,
    pl_agg_multi="mean",
    log_stats=True,
):
    if pseudo_csv_path is None:
        return []
    p = Path(pseudo_csv_path)
    if not p.is_absolute():
        p = data_root / p
    if not p.is_file():
        return []

    labeled_path = Path(labeled_csv_path)
    if not labeled_path.is_absolute():
        labeled_path = data_root / labeled_path
    labeled_df = pd.read_csv(labeled_path) if labeled_path.is_file() else pd.DataFrame(columns=["filename", "start"])
    labeled_keys = {(str(r["filename"]), float(time_hms_to_seconds(r["start"]))) for _, r in labeled_df.iterrows()}

    df = pd.read_csv(p)
    n_rows_initial = len(df)
    if pl_filter and "primary_label_prob" in df.columns:
        df = df[df["primary_label_prob"] > float(pl_filter_thr)].copy()
    n_rows_after_filter = len(df)

    class_cols = [c for c in label_to_idx.keys() if c in df.columns]
    n_zeroed_values = 0
    if pl_zero_unconf and class_cols:
        thr = float(pl_zero_unconf_thr)
        n_zeroed_values = int((df.loc[:, class_cols] < thr).to_numpy().sum())
        df.loc[:, class_cols] = df.loc[:, class_cols].where(df.loc[:, class_cols] >= thr, 0.0)

    base = data_root / subdir
    n_cls = len(label_to_idx)
    n_bins = max(1, round(float(chunk_duration_s) / float(label_bin_s)))

    # map (filename, start_sec_5s) -> soft vector
    pseudo_map = {}
    n_overlap_with_labeled = 0
    for _, row in df.iterrows():
        parsed = parse_row_id_window(str(row["row_id"]))
        if parsed is None:
            continue
        fn, t0_s = parsed
        key = (fn, float(t0_s))
        if key in labeled_keys:
            n_overlap_with_labeled += 1
            continue
        v = np.zeros(n_cls, dtype=np.float32)
        for lab, idx in label_to_idx.items():
            if lab in row.index:
                try:
                    v[idx] = float(row[lab])
                except Exception:
                    v[idx] = 0.0
        pseudo_map[key] = v

    agg_mode = str(pl_agg_multi).strip().lower()
    if agg_mode not in {"mean", "max"}:
        raise ValueError(f"Unsupported pl_agg_multi={pl_agg_multi!r}. Use 'mean' or 'max'.")

    files = sorted({k[0] for k in pseudo_map.keys()})
    items = []
    step = float(label_bin_s)
    chunk_s = float(chunk_duration_s)
    eps = 1e-6
    for fn in files:
        path = base / fn
        if not path.is_file():
            continue
        info = sf.info(str(path))
        duration = float(info.frames) / float(info.samplerate)
        t_win = 0.0
        while t_win + chunk_s <= duration + eps:
            bin_vecs = []
            has_any = False
            for i in range(n_bins):
                t_bin = float(round(t_win + i * step, 6))
                vv = pseudo_map.get((fn, t_bin))
                if vv is None:
                    vv = np.zeros(n_cls, dtype=np.float32)
                else:
                    has_any = True
                bin_vecs.append(vv)
            if has_any:
                bin_arr = np.stack(bin_vecs, axis=0)  # (k, C)
                if agg_mode == "max":
                    soft = bin_arr.max(axis=0)
                else:
                    soft = bin_arr.mean(axis=0)
                items.append(
                    DatasetItem(
                        source="train_soundscapes_pseudo",
                        path=path,
                        primary_labels=(),
                        secondary_labels=(),
                        t0_sec=float(t_win),
                        soft_target=tuple(float(x) for x in soft.tolist()),
                        bin_soft_targets=tuple(tuple(float(x) for x in row) for row in bin_arr.tolist()),
                        is_pseudo=True,
                    )
                )
            t_win += step
    if log_stats:
        print(
            "[pseudo] "
            f"rows_initial={n_rows_initial} "
            f"rows_after_pl_filter={n_rows_after_filter} "
            f"rows_dropped_labeled_overlap={n_overlap_with_labeled} "
            f"class_values_zeroed={n_zeroed_values} "
            f"final_pseudo_items={len(items)}"
        )
    return items


class BirdClefTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg = cfg
        self.ds_cfg = cfg["dataset"]
        self.data_root = Path(cfg["data_root"])
        self.audio_to_spec = AudioToSpec(cfg)
        self.sample_rate = self.ds_cfg["sample_rate"]
        self.chunk_duration_s = self.ds_cfg["chunk_duration_s"]
        self.soundscape_label_bin_s = self.ds_cfg["soundscape_label_bin_s"]
        self.chunk_samples = round(self.chunk_duration_s * self.sample_rate)
        self.n_soundscape_bins = max(1, round(self.chunk_duration_s / self.soundscape_label_bin_s))
        self.crop_mode = CropMode(self.ds_cfg["crop_mode"])
        self.padding_mode = PaddingMode(self.ds_cfg["padding_mode"])
        self.train = cfg["is_train"]
        self.distill_perch = bool(cfg.get("distill_perch", False))
        self.secondary_label_weight = self.ds_cfg["secondary_label_weight"]
        self.label_smoothing = float(self._cfg.get("label_smoothing", 0.0) or 0.0)
        self.wave_level_online_aug = cfg["online_aug"]["wave_level"]
        self.use_reshape_waves_not_melspec = self.ds_cfg["use_reshape_waves_not_melspec"]
        self.wave_reshape_width = self.ds_cfg["wave_reshape_width"]
        self.rng = torch.Generator()
        self.rng.manual_seed(cfg["seed"])
        self.is_reversed_audio = cfg["is_reversed_audio"]
        self.do_peak_norm = bool(self.ds_cfg.get("do_peak_norm", False))
        self.train_audio_min_duration_s = float(self.ds_cfg.get("train_audio_min_duration_s", 0.0))
        self.return_soundscape_bin_targets = self.chunk_duration_s > self.soundscape_label_bin_s
        self.use_multicontext_stacked_specs = (
            str(cfg["model"].get("model_type", "sed")) == "multised_trans"
            and bool(cfg["model"].get("multicontext", False))
            and self.n_soundscape_bins > 1
            and not self.wave_level_online_aug
            and not self.use_reshape_waves_not_melspec
        )
        self.use_chunked_multicontext_specs = (
            bool(cfg["model"].get("chunked_multicontext", False))
            and self.n_soundscape_bins > 1
            and not self.wave_level_online_aug
            and not self.use_reshape_waves_not_melspec
        )

        wave_aug = cfg["wave_aug"]
        self.bg_noise_prob = wave_aug["background_noise_prob"]
        noise_dirs = wave_aug["background_noise_dirs"]
        self.bg_noise_min_snr = wave_aug["background_noise_min_snr_db"]
        self.bg_noise_max_snr = wave_aug["background_noise_max_snr_db"]
        self.awgn_prob = float(wave_aug.get("gaussian_noise_prob", 0.0))
        self.awgn_min_snr_db = float(wave_aug.get("gaussian_noise_min_snr_db", 10.0))
        self.awgn_max_snr_db = float(wave_aug.get("gaussian_noise_max_snr_db", 30.0))
        self.wave_gain_prob = float(wave_aug.get("random_gain_prob", 0.0))
        self.wave_gain_min_db = float(wave_aug.get("random_gain_min_db", -6.0))
        self.wave_gain_max_db = float(wave_aug.get("random_gain_max_db", 6.0))
        self.noise_paths = []
        if self.train and self.bg_noise_prob > 0:
            raw_paths = collect_noises(
                self.data_root,
                noise_dirs,
                filter_invalid_audio=bool(self.ds_cfg.get("filter_invalid_background_noise", False)),
            )
            min_bg = float(wave_aug.get("min_bg_noise_duration", 5.0))
            self.noise_paths, n_drop_dur = filter_noise_paths_min_duration(raw_paths, min_bg)
            print(
                "[dataset] background_noise: "
                f"min_duration_s={min_bg:g}, "
                f"kept {len(self.noise_paths)}/{len(raw_paths)}, "
                f"dropped_shorter_or_invalid={n_drop_dur}",
                flush=True,
            )
            if not self.noise_paths:
                print(
                    "[dataset] background_noise: WARNING no noise files left after min-duration filter",
                    flush=True,
                )

        self.label_to_idx = build_merged_label_to_idx(self.data_root, self.ds_cfg)
        self.num_classes = len(self.label_to_idx)

        self.items = []

        _train_specs = train_audio_csv_specs(self.ds_cfg)
        if _train_specs:
            for csv_rel, sub in _train_specs:
                self.index_train_audio(self.data_root / csv_rel, sub)
        if self.ds_cfg["use_train_soundscapes"]:
            self.index_soundscapes(
                self.data_root / self.ds_cfg["soundscapes_labels_csv"], self.ds_cfg["train_soundscapes_subdir"]
            )
        self.extra_sources_stats = {"empty": True}
        self.index_extra_sources()
        if self.ds_cfg["use_train_soundscapes"]:
            self.index_soundscape_pseudos()

        self.train_audio_dir = self.data_root / self.ds_cfg["train_audio_subdir"]
        self.replace_corrupted = self.ds_cfg["replace_corrupted"]
        self.train_audio_replacement = {}
        if self.replace_corrupted and self.ds_cfg["use_train_audio"]:
            red = self.data_root / "redownloaded_corrupted"
            if red.is_dir():
                seen_paths = set()
                for it in self.items:
                    if it.source != "train_audio":
                        continue
                    p = it.path
                    if p in seen_paths:
                        continue
                    seen_paths.add(p)
                    try:
                        rel = p.relative_to(self.train_audio_dir)
                    except ValueError:
                        continue
                    alt = red / rel
                    if alt.is_file():
                        self.train_audio_replacement[p] = alt
            n = len(self.train_audio_replacement)
            self.corrupted_replace_stats = {"replaced": n}
            print(f"[dataset] replace_corrupted: will be replaced {n} files")
        else:
            self.corrupted_replace_stats = {"replaced": 0}

        bl = load_merged_audio_blocklists(self.data_root, self.ds_cfg)
        if bl:
            before_items = len(self.items)
            kept = []
            for it in self.items:
                r = item_rel_under_data_root(it, self.data_root)
                if r is not None and r in bl:
                    continue
                kept.append(it)
            self.items = kept
            print(f"[dataset] audio_blocklist(s): dropped {before_items - len(self.items)} items (n_block={len(bl)})")

        self.remove_voices = bool(self.ds_cfg.get("remove_voices", False))
        self.remove_voices_proba = float(self.ds_cfg.get("remove_voices_proba", 1.0))
        self.remove_voices_sources: frozenset[str] = _normalize_remove_voices_sources(
            self.ds_cfg.get("remove_voices_sources")
        )
        self.log_voice_crop_fallback = bool(self.ds_cfg.get("log_voice_crop_fallback", True))
        self.log_voice_crop_fallback_dedupe = bool(self.ds_cfg.get("log_voice_crop_fallback_dedupe", True))
        self.voice_crop_fallback_minimize_overlap = bool(self.ds_cfg.get("voice_crop_fallback_minimize_overlap", True))
        self.voice_crop_fallback_skip_file = bool(self.ds_cfg.get("voice_crop_fallback_skip_file", False))
        self._voice_crop_fallback_seen: set[str] = set()
        self._voice_regions_by_rel: dict[str, list[tuple[float, float]]] = {}
        if self.remove_voices:
            self._load_voice_region_csvs()
            if self.voice_crop_fallback_skip_file:
                before_skip = len(self.items)
                self.items = [it for it in self.items if not self._item_requires_voice_crop_fallback_drop(it)]
                n_skip_fb = before_skip - len(self.items)
                if n_skip_fb:
                    print(
                        "[dataset] voice_crop_fallback_skip_file: "
                        f"dropped {n_skip_fb} items (no voice-safe crop at chunk={self.chunk_samples} samples, "
                        f"crop_mode={self.crop_mode.name})",
                        flush=True,
                    )

    def _load_voice_region_csvs(self) -> None:
        vr_dir = self.data_root / "voice_regions"
        loaded_csv = 0
        for key in _VOICE_REGION_CSV_STEMS:
            p = vr_dir / f"{key}_voice_regions.csv"
            if not p.is_file():
                continue
            loaded_csv += 1
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                rel = str(row["file_path"]).strip()
                if not rel:
                    continue
                self._voice_regions_by_rel[rel] = _parse_voice_regions_cell(row["voice_regions"])
        st = _voice_regions_index_stats(self._voice_regions_by_rel)
        if int(st["segments_total"]) == 0:
            dur_s = "seg_dur_s=(none)"
        else:
            dur_s = (
                f"seg_dur_s min={float(st['seg_dur_s_min']):.4g} "
                f"max={float(st['seg_dur_s_max']):.4g} "
                f"mean={float(st['seg_dur_s_mean']):.4g}"
            )
        print(
            "[dataset] remove_voices=True: "
            f"voice_regions_csv_files={loaded_csv}/{len(_VOICE_REGION_CSV_STEMS)}, "
            f"indexed_paths={st['indexed_paths']}, "
            f"files_with_voice={st['files_with_voice']}, "
            f"segments_total={st['segments_total']}, "
            f"{dur_s}"
        )
        rvs = sorted(self.remove_voices_sources)
        print(
            "[dataset] remove_voices_sources: "
            f"{rvs if rvs else []}" + (" (no buckets — voice masking disabled for all items)" if not rvs else ""),
            flush=True,
        )
        print(f"[dataset] remove_voices_proba={self.remove_voices_proba} (train: avoid-speech crop with this prob)", flush=True)

    def _voice_remove_policy_key(self, item: DatasetItem) -> str | None:
        if item.source == "train_audio":
            if item.path in self.train_audio_replacement:
                return "redownloaded_corrupted"
            return "train"
        b = _voice_remove_bucket(item.source)
        if b is None:
            return None
        if b == "train_audio":
            return "train"
        return b

    def _voice_rel_key_for_item(self, item: DatasetItem) -> str | None:
        rel = item_rel_under_data_root(item, self.data_root)
        if rel is None:
            return None
        if item.source == "train_audio" and item.path in self.train_audio_replacement:
            alt = self.train_audio_replacement[item.path]
            try:
                return alt.resolve().relative_to(self.data_root.resolve()).as_posix()
            except ValueError:
                pass
        return rel

    def _voice_regions_for_item(self, item: DatasetItem) -> list[tuple[float, float]] | None:
        if not self.remove_voices:
            return None
        pk = self._voice_remove_policy_key(item)
        if pk is None or pk not in self.remove_voices_sources:
            return None
        key = self._voice_rel_key_for_item(item)
        if key is None:
            return None
        return self._voice_regions_by_rel.get(key, [])

    def _item_requires_voice_crop_fallback_drop(self, item: DatasetItem) -> bool:
        if not self.remove_voices or not self.voice_crop_fallback_skip_file:
            return False
        pk = self._voice_remove_policy_key(item)
        if pk is None or pk not in self.remove_voices_sources:
            return False
        key = self._voice_rel_key_for_item(item)
        if key is None:
            return False
        vr = self._voice_regions_by_rel.get(key, [])
        if not vr:
            return False

        path = item.path
        if item.source == "train_audio" and path in self.train_audio_replacement:
            path = self.train_audio_replacement[path]
        if not path.is_file():
            return True
        try:
            info = sf.info(str(path))
        except Exception:
            return True
        n = round(float(info.frames) * float(self.sample_rate) / float(info.samplerate))
        chunk = int(self.chunk_samples)
        if n < chunk:
            return True
        max_start = n - chunk
        bad = _bad_sample_intervals_from_voice(vr, n, self.sample_rate)
        forb = _forbidden_start_intervals(bad, n, chunk)
        good = _complement_intervals(forb, 0, max_start)
        if not good:
            return True

        mode = self.crop_mode if self.train else CropMode.RMS
        if mode == CropMode.START_RANDOM:
            hi_r = min(2 * chunk, max_start)
            good = _intersect_start_intervals(good, 0, hi_r)
        elif mode == CropMode.END_RANDOM:
            lo_r = max(0, max_start - 2 * chunk)
            good = _intersect_start_intervals(good, lo_r, max_start)

        return not good

    def smooth_target(self, tgt: torch.Tensor) -> torch.Tensor:
        ls = self.label_smoothing
        if ls <= 0.0:
            return tgt
        num_classes = tgt.shape[-1]
        if num_classes <= 0:
            return tgt
        return tgt * (1.0 - ls) + ls * (tgt.sum(dim=-1, keepdim=True) / float(num_classes))

    def index_train_audio(self, csv_path, subdir):
        df = pd.read_csv(csv_path)
        base = self.data_root / subdir
        n_skip_short = 0
        for _, row in df.iterrows():
            path = base / str(row["filename"])
            if path.is_file():
                try:
                    info = sf.info(str(path))
                    dur_s = float(info.frames) / float(info.samplerate)
                    if dur_s < self.train_audio_min_duration_s:
                        n_skip_short += 1
                        continue
                except Exception:
                    pass
            prim = norm_class_label(row["primary_label"])
            sec_raw = parse_list_cell(row["secondary_labels"])
            sec_in_tax = filter_taxonomy_labels(sec_raw, self.label_to_idx)
            sec_in_tax = [s for s in sec_in_tax if s != prim]
            self.items.append(
                DatasetItem(
                    source="train_audio",
                    path=path,
                    primary_labels=(prim,),
                    secondary_labels=tuple(sec_in_tax),
                )
            )
        if n_skip_short and bool(self._cfg.get("is_train", True)):
            print(
                f"[dataset] train_audio: skipped {n_skip_short} csv rows "
                f"(file shorter than {self.train_audio_min_duration_s:g}s)"
            )

    def index_soundscapes(self, csv_path, subdir):
        extra = build_soundscape_items(
            self.data_root,
            csv_path,
            subdir,
            self.label_to_idx,
            self.chunk_duration_s,
            self.soundscape_label_bin_s,
        )
        self.items.extend(extra)

    def index_extra_sources(self):
        ed = self.ds_cfg.get("extra_sources_data") or []
        em = self.ds_cfg.get("extra_sources_meta") or []
        if not ed or not em:
            return

        from src.extra_sources import build_extra_source_items  # circular import fix

        report = bool(self._cfg.get("is_train", True))
        items, stats = build_extra_source_items(self._cfg, self.label_to_idx, log=report)
        self.items.extend(items)
        self.extra_sources_stats = stats

    def index_soundscape_pseudos(self):
        pl_path = self.ds_cfg.get("pl_path")
        if not pl_path:
            return
        extra = build_soundscape_pseudo_items(
            self.data_root,
            pl_path,
            self.data_root / self.ds_cfg["soundscapes_labels_csv"],
            self.ds_cfg["train_soundscapes_subdir"],
            self.label_to_idx,
            self.chunk_duration_s,
            self.soundscape_label_bin_s,
            pl_filter=bool(self.ds_cfg.get("pl_filter", False)),
            pl_filter_thr=float(self.ds_cfg.get("pl_filter_thr", 0.5)),
            pl_zero_unconf=bool(self.ds_cfg.get("pl_zero_unconf", False)),
            pl_zero_unconf_thr=float(self.ds_cfg.get("pl_zero_unconf_thr", 0.1)),
            pl_agg_multi=str(self.ds_cfg.get("pl_agg_multi", "mean")),
        )
        self.items.extend(extra)

    def encode_labels(self, primary, secondary):
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for p in primary:
            if p in self.label_to_idx:
                y[self.label_to_idx[p]] = 1.0
        w = float(self.secondary_label_weight)
        for s in secondary:
            if s in self.label_to_idx:
                i = self.label_to_idx[s]
                y[i] = max(float(y[i]), w)
        return y

    def encode_primary_only_labels(self, primary):
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        for p in primary:
            if p in self.label_to_idx:
                y[self.label_to_idx[p]] = 1.0
        return y

    def load_waveform(self, item):
        if item.source == "train_soundscapes":
            offset = round(float(item.t0_sec) * self.sample_rate)
            wav, sr = load_audio_soundfile(item.path, frame_offset=offset, num_frames=self.chunk_samples)
        else:
            path = item.path
            if item.source == "train_audio" and self.train_audio_replacement:
                path = self.train_audio_replacement.get(path, path)
            wav, sr = load_audio_soundfile(path)

        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        is_train_like = item.source == "train_audio" or str(item.source).startswith("extra_")
        if is_train_like:
            mode = self.crop_mode if self.train else CropMode.RMS
            vr = self._voice_regions_for_item(item)
            if vr is not None:
                use_avoid_speech = True
                if self.train and self.remove_voices_proba < 1.0:
                    use_avoid_speech = torch.rand((), generator=self.rng).item() < self.remove_voices_proba

                if use_avoid_speech:
                    voice_fallback_log = None
                    if self.log_voice_crop_fallback:

                        def voice_fallback_log(reason: str, voice_extras: dict[str, Any] | None = None) -> None:
                            rel = item_rel_under_data_root(item, self.data_root) or str(item.path)
                            dedupe_key = f"{reason}|{rel}|{self.chunk_samples}"
                            if self.log_voice_crop_fallback_dedupe and dedupe_key in self._voice_crop_fallback_seen:
                                return
                            self._voice_crop_fallback_seen.add(dedupe_key)
                            dur = float(wav.shape[1]) / float(self.sample_rate)
                            payload: dict[str, Any] = {
                                "reason": reason,
                                "audio": rel,
                                "primary_labels": list(item.primary_labels),
                                "duration_s": round(dur, 4),
                                "voice_regions_sec": [[float(t0), float(t1)] for t0, t1 in vr],
                                "chunk_samples": int(self.chunk_samples),
                                "crop_mode": mode.name,
                            }
                            if voice_extras:
                                payload.update(voice_extras)
                            print(
                                "[dataset] voice_crop_fallback: " + json.dumps(payload, ensure_ascii=False),
                                flush=True,
                            )

                    wav = crop_audio_avoid_voice_regions(
                        wav,
                        self.chunk_samples,
                        vr,
                        self.sample_rate,
                        mode,
                        self.rng,
                        self.padding_mode,
                        self.train,
                        voice_fallback_log=voice_fallback_log,
                        voice_fallback_minimize_overlap=self.voice_crop_fallback_minimize_overlap,
                    )
                else:
                    wav = crop_audio(wav, self.chunk_samples, mode, self.rng, self.padding_mode, self.train)
            else:
                wav = crop_audio(wav, self.chunk_samples, mode, self.rng, self.padding_mode, self.train)
        else:
            c = self.chunk_samples
            t = wav.shape[1]
            if t < c:
                wav = pad_waveform_to_length(wav, c, self.padding_mode, self.rng, self.train)
            elif t > c:
                wav = wav[:, :c]

        if self.is_reversed_audio:
            wav = torch.flip(wav, dims=(1,))

        return wav

    def add_background_noise(self, wav):
        if (
            not self.train
            or self.bg_noise_prob <= 0
            or not self.noise_paths
            or torch.rand((), generator=self.rng).item() >= self.bg_noise_prob
        ):
            return wav
        path = self.noise_paths[int(torch.randint(0, len(self.noise_paths), (1,), generator=self.rng).item())]
        noise, sr = load_audio_soundfile(path)
        if noise.shape[0] > 1:
            noise = noise.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            noise = torchaudio.functional.resample(noise, sr, self.sample_rate)
        length = wav.shape[1]
        noise = prepare_noise_tensor(noise, length, self.rng)
        noise_e2 = torch.linalg.vector_norm(noise, ord=2, dim=-1).pow(2).clamp(min=0.0)
        sig_e2 = torch.linalg.vector_norm(wav, ord=2, dim=-1).pow(2).clamp(min=0.0)
        if float(noise_e2.item()) < 1e-18 or (float(sig_e2.item()) < 1e-18 and float(noise_e2.item()) < 1e-18):
            print("noise or signal is too small")
            return wav
        lo, hi = self.bg_noise_min_snr, self.bg_noise_max_snr
        if hi < lo:
            lo, hi = hi, lo
        u = torch.rand((), generator=self.rng).item()
        snr_db = lo + u * (hi - lo)
        snr = torch.tensor([snr_db], dtype=torch.float32)
        return torchaudio.functional.add_noise(wav, noise, snr)

    def add_random_gain_wave(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.train:
            return wav
        return apply_wave_random_gain_db(
            wav, self.wave_gain_min_db, self.wave_gain_max_db, self.wave_gain_prob, generator=self.rng
        )

    def add_awgn(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.train:
            return wav
        return apply_wave_awgn_snr(wav, self.awgn_min_snr_db, self.awgn_max_snr_db, self.awgn_prob, generator=self.rng)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        max_tries = 8
        for attempt in range(max_tries):
            try:
                item = self.items[idx]
                wav = self.load_waveform(item)
                wav = self.add_background_noise(wav)
                wav = self.add_random_gain_wave(wav)
                wav = self.add_awgn(wav)
                if self.do_peak_norm:
                    wav = peak_normalize_waveform(wav)
                if item.soft_target is not None:
                    target_raw = torch.tensor(item.soft_target, dtype=torch.float32)
                else:
                    target_raw = self.encode_labels(item.primary_labels, item.secondary_labels)
                target = self.smooth_target(target_raw)
                y_bins = None
                if self.return_soundscape_bin_targets:
                    if item.source == "train_soundscapes":
                        bins = item.bin_primary_labels or tuple(() for _ in range(self.n_soundscape_bins))
                        if len(bins) < self.n_soundscape_bins:
                            bins = tuple(bins) + tuple(() for _ in range(self.n_soundscape_bins - len(bins)))
                        elif len(bins) > self.n_soundscape_bins:
                            bins = tuple(bins[: self.n_soundscape_bins])
                        y_bins = torch.stack([self.encode_primary_only_labels(b) for b in bins], dim=0)
                    elif item.source == "train_soundscapes_pseudo" and item.bin_soft_targets is not None:
                        bins_soft = item.bin_soft_targets
                        if len(bins_soft) < self.n_soundscape_bins:
                            bins_soft = tuple(bins_soft) + tuple(
                                tuple(0.0 for _ in range(self.num_classes))
                                for _ in range(self.n_soundscape_bins - len(bins_soft))
                            )
                        elif len(bins_soft) > self.n_soundscape_bins:
                            bins_soft = tuple(bins_soft[: self.n_soundscape_bins])
                        y_bins = torch.tensor(bins_soft, dtype=torch.float32)
                    else:
                        y_bins = target_raw.unsqueeze(0).repeat(self.n_soundscape_bins, 1)
                    y_bins = self.smooth_target(y_bins)
                if self.wave_level_online_aug:
                    if y_bins is not None:
                        return wav, target, y_bins, torch.tensor(item.is_pseudo, dtype=torch.bool)
                    return wav, target, torch.tensor(item.is_pseudo, dtype=torch.bool)
                if self.use_multicontext_stacked_specs or self.use_chunked_multicontext_specs:
                    t = wav.shape[1]
                    edges = torch.linspace(0, t, steps=self.n_soundscape_bins + 1).round().long()
                    specs = []
                    for bi in range(self.n_soundscape_bins):
                        l, r = int(edges[bi]), int(edges[bi + 1])
                        if r <= l:
                            r = min(l + 1, t)
                        seg = wav[:, l:r]
                        specs.append(self.audio_to_spec(seg, self.train))
                    spec = torch.stack(specs, dim=0)
                elif self.use_reshape_waves_not_melspec:
                    spec = reshape_waveform(wav, self.wave_reshape_width)
                else:
                    spec = self.audio_to_spec(wav, self.train)
                if y_bins is not None:
                    if self.distill_perch:
                        return spec, target, y_bins, torch.tensor(item.is_pseudo, dtype=torch.bool), wav
                    return spec, target, y_bins, torch.tensor(item.is_pseudo, dtype=torch.bool)
                if self.distill_perch:
                    return spec, target, torch.tensor(item.is_pseudo, dtype=torch.bool), wav
                return spec, target, torch.tensor(item.is_pseudo, dtype=torch.bool)
            except Exception as e:
                if attempt + 1 >= max_tries:
                    raise RuntimeError(f"failed to load sample after {max_tries} attempts, last {idx=}") from e
                idx = int(torch.randint(0, len(self.items), (1,), generator=self.rng).item())

    @property
    def mel_fn(self):
        return self.audio_to_spec


def build_dataloader(cfg: dict, dataset, sampler=None, *, train=True):
    nw = cfg["num_workers"]
    kw = {
        "num_workers": nw,
        "pin_memory": torch.cuda.is_available(),
    }
    if nw > 0:
        kw["persistent_workers"] = True
    batch_sampler = None
    if train:
        batch_sampler = get_pseudo_balanced_batch_sampler(cfg, dataset)
        if batch_sampler is None:
            batch_sampler = get_source_balanced_batch_sampler(cfg, dataset)
    if batch_sampler is not None:
        if sampler is not None:
            raise ValueError("batch sampler cannot be combined with cfg.sampler != 'none'")
        kw["batch_sampler"] = batch_sampler
        return DataLoader(dataset, **kw)
    kw["batch_size"] = cfg["bs"]
    if sampler is not None:
        kw["sampler"] = sampler
        kw["shuffle"] = False
    else:
        kw["shuffle"] = train
    kw["drop_last"] = train
    return DataLoader(dataset, **kw)


def build_train_val_dataloaders(cfg: dict, train_dataset, val_dataset):
    train_sampler = get_sampler(cfg, train_dataset)
    train_loader = build_dataloader(cfg, train_dataset, train_sampler, train=True)
    val_loader = build_dataloader(cfg, val_dataset, None, train=False)
    return train_loader, val_loader
