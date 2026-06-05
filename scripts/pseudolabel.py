import argparse
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy import ndimage
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lightning import build_model  # noqa: E402
from src.sed_infer_runtime import INFER_S, NUM_SEGMENTS, MelRuntime, infer_stem_probs, parse_row_id  # noqa: E402
from src.utils import cfg_to_dict  # noqa: E402

DEFAULT_WEIGHT_DIRS = [
    ROOT / "weights" / "0_919_repeated_full",
    ROOT / "weights" / "0_920_best_full_hgnetv2_b3_ssld_stage2_ft_in1k",
    ROOT / "weights" / "0_920_best_full_tf_efficientnet_b3_ns_jft_in1k_higher_lr",
    ROOT / "weights" / "0_920_best_full_tf_efficientnetv2_s_in21k_ft_in1k",
]


def load_config(path: Path) -> dict:
    path = path.resolve()
    spec = importlib.util.spec_from_file_location("_pseudo_cfg_", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return cfg_to_dict(mod.cfg)


def resolve_model_paths(weights_dir: Path) -> tuple[Path, Path]:
    weights_dir = weights_dir.resolve()
    exp = weights_dir.name
    ckpt = weights_dir / f"{exp}_soup_k3_avg.ckpt"
    if not ckpt.is_file():
        candidates = sorted(weights_dir.glob("*soup*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No soup checkpoint in {weights_dir}")
        ckpt = candidates[-1]

    config = ROOT / "configs" / f"{exp}.py"
    if not config.is_file():
        run_json = weights_dir / "fold0_run.json"
        if run_json.is_file():
            config = Path(json.loads(run_json.read_text())["config_path"])
    if not config.is_file():
        raise FileNotFoundError(f"Config not found for {weights_dir}")
    return config, ckpt


def load_lightning_sed_weights(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    pref = "model."
    sd = {
        k[len(pref) :]: v
        for k, v in raw.items()
        if k.startswith(pref) and not k.startswith(pref + "criterion.")
    }
    mine = model.state_dict()
    missing = sorted(k for k in set(mine) - set(sd) if not k.startswith("criterion."))
    mismatched = sorted(k for k in set(mine) & set(sd) if mine[k].shape != sd[k].shape)
    if missing:
        raise RuntimeError(f"Missing keys when loading {ckpt_path}: {missing[:20]}")
    if mismatched:
        raise RuntimeError(f"Shape mismatch when loading {ckpt_path}: {mismatched[:5]}")
    model.load_state_dict(sd, strict=False)


def load_ogg_mono(path: str) -> tuple[np.ndarray, int]:
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = np.mean(w, axis=1) if w.ndim == 2 and w.shape[1] > 1 else w.reshape(-1)
    return w, int(sr)


def build_train_index(data_root: Path, *, max_files: int | None) -> tuple[list[str], dict[str, str], dict[str, list[tuple[int, float]]]]:
    train_dir = data_root / "train_soundscapes"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train_soundscapes dir: {train_dir}")

    audio_files = sorted(train_dir.glob("*.ogg"))
    if max_files is not None:
        audio_files = audio_files[: int(max_files)]

    row_ids: list[str] = []
    stem_to_path: dict[str, str] = {}
    stem_to_rows: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for audio_p in audio_files:
        stem = audio_p.stem
        stem_to_path[stem] = str(audio_p)
        for end_sec in range(int(INFER_S), int(NUM_SEGMENTS * INFER_S) + 1, int(INFER_S)):
            rid = f"{stem}_{end_sec}"
            row_ids.append(rid)
            stem_to_rows[stem].append((len(row_ids) - 1, float(end_sec)))

    return row_ids, stem_to_path, stem_to_rows


def apply_power_to_low_ranked_cols(p: np.ndarray, top_k: int = 30, exponent: float = 2.0) -> np.ndarray:
    p = p.copy()
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p


def gaussian_smooth_probs(probs: np.ndarray, kernel: np.ndarray | None = None) -> np.ndarray:
    if kernel is None:
        kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float32)
    out = np.zeros_like(probs)
    for class_idx in range(probs.shape[1]):
        out[:, class_idx] = ndimage.convolve1d(probs[:, class_idx], kernel, mode="nearest")
    return out


def postprocessing_soundscapes(probs: np.ndarray, top: int = 1) -> np.ndarray:
    n, f = probs.shape
    if n % NUM_SEGMENTS != 0:
        raise ValueError(f"Row count {n} is not divisible by {NUM_SEGMENTS}")
    x = probs.reshape((n // NUM_SEGMENTS, NUM_SEGMENTS, f))
    mean_ = np.mean(np.sort(x, axis=1)[:, -top:], axis=1, keepdims=True)
    x = x * mean_
    return x.reshape((n, f))


def run_model_probs(
    *,
    config_path: Path,
    ckpt_path: Path,
    device: torch.device,
    waves: dict[str, np.ndarray],
    sr_by_stem: dict[str, int],
    stem_to_rows: dict[str, list[tuple[int, float]]],
    col_to_model: np.ndarray,
    n_rows: int,
    n_labels: int,
    is_reversed_audio: bool,
) -> np.ndarray:
    cfg = load_config(config_path)
    cfg["model"]["backbone"]["pretrained"] = False
    cfg["model"]["backbone"]["init_checkpoint"] = None

    model = build_model(cfg).eval().to(device)
    load_lightning_sed_weights(model, ckpt_path)
    mel_runtime = MelRuntime(cfg)

    probs = np.zeros((n_rows, n_labels), dtype=np.float32)
    for stem in tqdm(sorted(stem_to_rows.keys()), desc=f"infer {ckpt_path.parent.name}"):
        w = waves[stem]
        if is_reversed_audio:
            w = np.ascontiguousarray(w[::-1])
        infer_stem_probs(
            model,
            mel_runtime,
            w,
            sr_by_stem[stem],
            device=device,
            col_to_model=col_to_model,
            row_indices=stem_to_rows[stem],
            out_probs=probs,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate train_soundscapes pseudolabels")
    p.add_argument("--data-root", type=Path, default=ROOT / "data")
    p.add_argument(
        "--weights-dirs",
        type=Path,
        nargs="+",
        default=DEFAULT_WEIGHT_DIRS,
        help="directories with soup checkpoints",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "TOP_SVALKA" / "pseudos_ensemble.csv",
    )
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: cuda if available)")
    p.add_argument("--max-files", type=int, default=None, help="debug: only first N soundscape files")
    p.add_argument("--n-jobs", type=int, default=None, help="parallel audio loading jobs")
    p.add_argument("--fold-id", type=int, default=-1, help="fold_id column value (reference uses -1)")
    p.add_argument("--power-top-k", type=int, default=30)
    p.add_argument("--power-exponent", type=float, default=2.0)
    p.add_argument("--post-top", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    n_jobs = args.n_jobs if args.n_jobs is not None else (os.cpu_count() or 8)

    tax = pd.read_csv(data_root / "taxonomy.csv")
    sub_template = pd.read_csv(data_root / "sample_submission.csv")
    label_cols = [c for c in sub_template.columns if c != "row_id"]
    tax_set = set(tax["primary_label"].astype(str))
    for c in label_cols:
        if str(c) not in tax_set:
            raise KeyError(f"Label {c} missing in taxonomy")

    label_to_idx = {str(lb): i for i, lb in enumerate(tax["primary_label"])}
    col_to_model = np.array([label_to_idx[str(c)] for c in label_cols], dtype=np.int64)

    row_ids, stem_to_path, stem_to_rows = build_train_index(data_root, max_files=args.max_files)
    n_rows = len(row_ids)
    n_labels = len(label_cols)
    print(f"train_soundscapes: {len(stem_to_path)} files, {n_rows} rows ({NUM_SEGMENTS} segments/file)")

    stems = sorted(stem_to_path.keys())

    def load_one(st: str) -> tuple[str, np.ndarray, int]:
        w, sr = load_ogg_mono(stem_to_path[st])
        return st, w, sr

    print(f"loading {len(stems)} audio files (n_jobs={n_jobs})...")
    loaded = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(load_one)(st) for st in tqdm(stems, desc="audio"))
    waves = {s: w for s, w, _ in loaded}
    sr_by_stem = {s: sr for s, _, sr in loaded}

    is_reversed = False
    model_dirs = [Path(d).resolve() for d in args.weights_dirs]
    ensemble_probs = np.zeros((n_rows, n_labels), dtype=np.float32)

    for weights_dir in model_dirs:
        config_path, ckpt_path = resolve_model_paths(weights_dir)
        cfg_probe = load_config(config_path)
        is_reversed = bool(cfg_probe.get("is_reversed_audio", False))
        print(f"\n=== {weights_dir.name} ===")
        print("config:", config_path)
        print("ckpt:", ckpt_path)
        model_probs = run_model_probs(
            config_path=config_path,
            ckpt_path=ckpt_path,
            device=device,
            waves=waves,
            sr_by_stem=sr_by_stem,
            stem_to_rows=stem_to_rows,
            col_to_model=col_to_model,
            n_rows=n_rows,
            n_labels=n_labels,
            is_reversed_audio=is_reversed,
        )
        ensemble_probs += model_probs

    ensemble_probs /= float(len(model_dirs))
    print("\npostprocessing: power adj -> gaussian -> soundscape max-mean")

    ensemble_probs = apply_power_to_low_ranked_cols(
        ensemble_probs,
        top_k=args.power_top_k,
        exponent=args.power_exponent,
    )
    ensemble_probs = gaussian_smooth_probs(ensemble_probs)
    ensemble_probs = postprocessing_soundscapes(ensemble_probs, top=args.post_top)

    best_idx = ensemble_probs.argmax(axis=1)
    primary_labels = [label_cols[i] for i in best_idx]
    primary_probs = ensemble_probs[np.arange(n_rows), best_idx]

    out_df = pd.DataFrame({"row_id": row_ids, "fold_id": int(args.fold_id)})
    out_df[label_cols] = ensemble_probs
    out_df["primary_label"] = primary_labels
    out_df["primary_label_prob"] = primary_probs

    ref_cols = list(pd.read_csv(ROOT / "TOP_SVALKA" / "top13_raw_pseudos_25april.csv", nrows=0).columns)
    out_df = out_df[ref_cols]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"saved {args.out_csv} ({len(out_df)} rows, {len(out_df.columns)} cols)")


if __name__ == "__main__":
    main()
