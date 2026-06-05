import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

AUDIO_SUFFIX = {
    ".ogg",
    ".mp3",
    ".wav",
    ".flac",
    ".opus",
    ".m4a",
    ".aac",
}

DEFAULT_SCAN_REL_ROOTS = [
    "birdclef-2022/train_audio",
    "birdclef-2023/train_audio",
    "birdclef-2024/train_audio",
    "birdclef-2025/train_audio",
    "cornell_1/train_audio",
    "cornell_2/train_audio",
    "xc_1/A-M",
    "xc_2/N-Z",
    "birdclef-2025-extra-data/birdclef2025_extra_species_data",
    "birdclef-2025-extra-data/birdclef2025_extra_target_data",
]


def discover_scan_rel_roots(data_root: Path, roots: list[str] | None = None) -> list[str]:
    data_root = Path(data_root).resolve()
    roots = list(roots or DEFAULT_SCAN_REL_ROOTS)
    out: list[str] = []
    seen: set[str] = set()
    for r in roots:
        r = r.replace("\\", "/")
        if r in seen:
            continue
        if (data_root / r).is_dir():
            seen.add(r)
            out.append(r)
            continue
        nested = f"old_extra_data/{r}"
        if nested not in seen and (data_root / nested).is_dir():
            seen.add(nested)
            out.append(nested)
    return out


def norm_label(s: str) -> str:
    return str(s).strip()


def infer_folder_primary_label(audio_path: Path, scan_root: Path) -> str | None:
    try:
        rel = audio_path.resolve().relative_to(scan_root.resolve())
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) < 2:
        root_name = scan_root.name.strip()
        return f"__root__{root_name}" if root_name else None
    parent_dir = parts[-2]
    if parent_dir.lower() == "data" and len(parts) >= 4:
        return norm_label(parts[-3])
    return norm_label(parent_dir) if parent_dir else None


def scan_anchor_roots(data_root: Path, rel_roots: list[str]):
    roots: list[tuple[str, Path]] = []
    for rel in rel_roots:
        anchor = data_root / rel.replace("\\", "/")
        if anchor.is_dir():
            roots.append((rel, anchor))
    return roots


def build_old_manifest(
    data_root: Path,
    *,
    out_manifest: Path,
    out_taxonomy: Path,
    out_stats_json: Path | None = None,
    scan_rel_roots: list[str] | None = None,
    validate_audio: bool = False,
    validate_rejected_log: Path | None = None,
    target_sr: int = 32000,
    validate_audio_workers: int | None = None,
) -> dict:
    data_root = data_root.resolve()
    rel_roots = scan_rel_roots or discover_scan_rel_roots(data_root)
    anchors = scan_anchor_roots(data_root, rel_roots)

    rows: list[tuple[str, str]] = []
    seen_rel: set[str] = set()
    skipped_no_class = 0
    anchors_used = []

    for rel_anchor, anchor in anchors:
        anchors_used.append(rel_anchor)
        for p in anchor.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in AUDIO_SUFFIX:
                continue
            label = infer_folder_primary_label(p, anchor)
            if label is None or label == ".":
                skipped_no_class += 1
                continue
            rel_file = rel_from_root(data_root, p)
            if rel_file in seen_rel:
                continue
            seen_rel.add(rel_file)
            rows.append((rel_file, label))

    rejected_validate = 0
    validate_audio_workers_used = 0
    if validate_audio:
        from tqdm import tqdm

        from src.audio_validate import VALIDATE_AUDIO_DEFAULT_WORKERS, validate_audio_file

        rels = sorted({r for r, _ in rows})
        bad: set[str] = set()
        nw = max(
            1,
            int(validate_audio_workers) if validate_audio_workers is not None else int(VALIDATE_AUDIO_DEFAULT_WORKERS),
        )
        validate_audio_workers_used = nw

        def _check(rel: str) -> tuple[str, bool]:
            ap = data_root / rel
            ok, _why = validate_audio_file(ap, target_sr=target_sr)
            return rel, ok

        with ThreadPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(_check, rel) for rel in rels]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="validate_audio"):
                rel, ok = fut.result()
                if not ok:
                    bad.add(rel)
        if bad:
            rejected_validate = len(bad)
            rows = [(r, lab) for r, lab in rows if r not in bad]
            log_p = validate_rejected_log or (out_manifest.parent / "pretrain_manifest_rejected_audio.txt")
            log_p.parent.mkdir(parents=True, exist_ok=True)
            log_p.write_text("\n".join(sorted(bad)) + "\n", encoding="utf-8")

    df_train = pd.DataFrame(
        {"filename": [r for r, _ in rows], "primary_label": [lab for _, lab in rows], "secondary_labels": "[]"}
    )
    df_train["_basename"] = df_train["filename"].map(lambda x: Path(str(x)).name)
    before_dedup = len(df_train)
    df_train = df_train.drop_duplicates(subset=["_basename", "primary_label"], keep="first").reset_index(drop=True)
    df_train = df_train.drop(columns=["_basename"])
    dedup_removed = before_dedup - len(df_train)

    classes_sorted = sorted(set(df_train["primary_label"].tolist()))
    class_to_ix = {c: i for i, c in enumerate(classes_sorted)}

    df_tax = pd.DataFrame(
        {
            "primary_label": classes_sorted,
            "inat_taxon_id": [class_to_ix[c] for c in classes_sorted],
            "scientific_name": classes_sorted,
            "common_name": classes_sorted,
            "class_name": ["pretrain_folder"] * len(classes_sorted),
        }
    )

    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_manifest, index=False)
    df_tax.to_csv(out_taxonomy, index=False)

    stats = {
        "anchors_used": anchors_used,
        "num_files": len(rows),
        "num_files_after_dedup": len(df_train),
        "dedup_removed": int(dedup_removed),
        "num_classes": len(classes_sorted),
        "skipped_parent_class": skipped_no_class,
        "missing_anchors": [r for r in rel_roots if r not in anchors_used],
        "rejected_validate_audio": int(rejected_validate),
        "validate_audio_workers": int(validate_audio_workers_used),
    }
    if out_stats_json:
        out_stats_json.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def rel_from_root(data_root: Path, absolute: Path) -> str:
    return str(absolute.resolve().relative_to(data_root.resolve()).as_posix())


def main():
    from src.audio_validate import VALIDATE_AUDIO_DEFAULT_WORKERS

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-manifest", type=Path, default=None)
    ap.add_argument("--out-taxonomy", type=Path, default=None)
    ap.add_argument("--out-stats", type=Path, default=None)
    ap.add_argument(
        "--validate-audio",
        action="store_true",
        help="drop rows that fail decode/ffmpeg sniff (see src.audio_validate)",
    )
    ap.add_argument("--validate-rejected-log", type=Path, default=None)
    ap.add_argument("--target-sr", type=int, default=32000)
    ap.add_argument(
        "--validate-audio-workers",
        type=int,
        default=VALIDATE_AUDIO_DEFAULT_WORKERS,
        help="thread pool size when --validate-audio is set",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_root = args.data_root or (root / "data" / "old_extra_data")
    out_manifest = args.out_manifest or (data_root / "pretrain_manifest.csv")
    out_taxonomy = args.out_taxonomy or (data_root / "pretrain_taxonomy.csv")
    out_stats = args.out_stats or (data_root / "pretrain_folder_stats.json")
    st = build_old_manifest(
        data_root,
        out_manifest=out_manifest,
        out_taxonomy=out_taxonomy,
        out_stats_json=out_stats,
        validate_audio=bool(args.validate_audio),
        validate_rejected_log=args.validate_rejected_log,
        target_sr=int(args.target_sr),
        validate_audio_workers=int(args.validate_audio_workers),
    )
    print(json.dumps(st, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
