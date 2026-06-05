import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_AUDIO_EXT = frozenset({".ogg", ".wav", ".mp3", ".m4a", ".flac", ".opus"})

VOICE_SOURCES = ("train_audio", "xc", "inat", "tsa", "redownloaded_corrupted")


def load_config(path: Path) -> dict:
    from src.utils import cfg_to_dict

    path = path.resolve()
    spec = importlib.util.spec_from_file_location("_voice_cfg_", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return cfg_to_dict(mod.cfg)


def _normalize_data_root(cfg: dict) -> Path:
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = (ROOT / data_root).resolve()
        cfg["data_root"] = str(data_root)
    return data_root


def _iter_audio_files_under(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in _AUDIO_EXT:
            out.append(p)
    return sorted(out)


def _rel_posix(data_root: Path, p: Path) -> str:
    return p.resolve().relative_to(data_root.resolve()).as_posix()


def collect_paths_train_audio(data_root: Path, ds: dict) -> list[Path]:
    sub = str(ds.get("train_audio_subdir", "train_audio") or "train_audio")
    return _iter_audio_files_under(data_root / sub)


def collect_paths_redownloaded_corrupted(data_root: Path) -> list[Path]:
    return _iter_audio_files_under(data_root / "redownloaded_corrupted")


def collect_paths_extra_tag(data_root: Path, cfg: dict, want_tag: str) -> list[Path]:
    from src.extra_source_tag import infer_extra_source_tag

    ds = cfg["dataset"]
    paths_data = ds.get("extra_sources_data") or []
    paths_meta = ds.get("extra_sources_meta") or []
    if len(paths_data) != len(paths_meta):
        raise ValueError("extra_sources_data / extra_sources_meta length mismatch")
    out: list[Path] = []
    for raw_d, raw_m in zip(paths_data, paths_meta, strict=False):
        pd_ = Path(raw_d)
        pm = Path(raw_m)
        tag = infer_extra_source_tag(pd_, pm)
        if tag != want_tag:
            continue
        root = pd_ if pd_.is_absolute() else (data_root / pd_)
        out.extend(_iter_audio_files_under(root))
    # uniq stable
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in out:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(Path(k))
    return sorted(uniq)


def run_source(
    *,
    source: str,
    paths: list[Path],
    data_root: Path,
    vad_bundle,
    out_csv: Path,
    resume: bool,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    from src.vad import MIN_VAD_AUDIO_DURATION_S, detect_voice_timestamps

    model, get_speech_timestamps = vad_bundle

    done: set[str] = set()
    existing_rows: list[dict] = []
    if resume and out_csv.is_file():
        old = pd.read_csv(out_csv)
        for _, row in old.iterrows():
            fp = str(row["file_path"]).strip()
            done.add(fp)
            existing_rows.append({"file_path": fp, "voice_regions": row["voice_regions"]})

    rows: list[dict] = list(existing_rows) if resume else []
    to_run = [p for p in paths if _rel_posix(data_root, p) not in done]
    n_skip = sum(1 for p in paths if _rel_posix(data_root, p) in done)
    for p in tqdm(
        to_run,
        desc=f"vad:{source}",
        unit="file",
        total=len(paths),
        initial=n_skip,
        mininterval=0.3,
    ):
        rel = _rel_posix(data_root, p)
        try:
            info = sf.info(str(p))
            dur = float(info.frames) / float(info.samplerate)
            if dur < MIN_VAD_AUDIO_DURATION_S:
                ts = []
            else:
                ts = detect_voice_timestamps(p, model, get_speech_timestamps)
        except Exception as e:
            tqdm.write(f"[warn] {rel}: {e}")
            ts = []
        if ts:
            tqdm.write(f"[voice] {rel} {json.dumps(ts, ensure_ascii=False)}")
        rows.append({"file_path": rel, "voice_regions": json.dumps(ts, ensure_ascii=False)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["file_path"], keep="last").sort_values("file_path")
    df.to_csv(out_csv, index=False)
    print(f"[voice_regions] wrote {len(df)} rows -> {out_csv}")


def main() -> None:
    from src.vad import default_silero_vad_repo, load_silero_vad

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--source",
        nargs="+",
        choices=[*VOICE_SOURCES, "all"],
        default=["all"],
        metavar="SRC",
    )
    ap.add_argument(
        "--vad-dir",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    ap.add_argument(
        "--use-gpu",
        action="store_true",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    src_arg = list(args.source)
    if len(src_arg) == 1 and src_arg[0] == "all":
        sources = list(VOICE_SOURCES)
    elif "all" in src_arg:
        ap.error("'all' must be the only value for --source (do not mix with xc/inat/...)")
    else:
        sources = src_arg

    import torch

    if args.use_gpu:
        device_mode = "cuda"
    else:
        device_mode = args.device
    if device_mode == "cpu":
        use_gpu = False
    elif device_mode == "cuda":
        use_gpu = torch.cuda.is_available()
        if not use_gpu:
            print(
                "[vad] --device cuda but torch.cuda.is_available() is False; using CPU",
                flush=True,
            )
    else:
        use_gpu = torch.cuda.is_available()

    cfg = load_config(args.config)
    data_root = _normalize_data_root(cfg)
    ds = cfg["dataset"]
    silero_repo = args.vad_dir if args.vad_dir is not None else default_silero_vad_repo(ROOT)
    model, get_speech_timestamps = load_silero_vad(repo_dir=silero_repo, use_gpu=use_gpu)
    try:
        vad_dev = next(model.parameters()).device
    except StopIteration:
        vad_dev = torch.device("cpu")
    print(f"[vad] Silero on {vad_dev}", flush=True)

    out_dir = data_root / "voice_regions"

    for src in sources:
        if src == "train_audio":
            paths = collect_paths_train_audio(data_root, ds)
        elif src == "redownloaded_corrupted":
            paths = collect_paths_redownloaded_corrupted(data_root)
        else:
            paths = collect_paths_extra_tag(data_root, cfg, src)
        if not paths:
            print(f"[voice_regions] skip {src}: no audio files")
            continue
        out_csv = out_dir / f"{src}_voice_regions.csv"
        run_source(
            source=src,
            paths=paths,
            data_root=data_root,
            vad_bundle=(model, get_speech_timestamps),
            out_csv=out_csv,
            resume=bool(args.resume),
        )


if __name__ == "__main__":
    main()
