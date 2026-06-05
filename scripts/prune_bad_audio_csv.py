import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_audio_path(data_root: Path, filename: str, path_prefix: str) -> Path:
    rel = (Path(path_prefix) / filename).as_posix() if path_prefix else str(filename)
    return (data_root / rel).resolve()


def main() -> None:
    from src.audio_validate import VALIDATE_AUDIO_DEFAULT_WORKERS, validate_audio_file

    ap = argparse.ArgumentParser(description="Prune CSV rows with corrupt/unreadable audio files")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True, help="CSV with a filename column")
    ap.add_argument(
        "--path-prefix",
        type=str,
        default="",
        help="subdir under data_root prepended to each filename (e.g. train_audio)",
    )
    ap.add_argument("--out-csv", type=Path, default=None, help="default: overwrite --csv (backup .bak)")
    ap.add_argument(
        "--reject-log",
        type=Path,
        default=None,
        help="relative paths (posix) one per line; default next to csv",
    )
    ap.add_argument("--target-sr", type=int, default=32000)
    ap.add_argument("--workers", type=int, default=VALIDATE_AUDIO_DEFAULT_WORKERS)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    csv_path = args.csv.resolve()
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise SystemExit("CSV must contain a filename column")

    rel_per_row: list[str | None] = []
    for _, row in df.iterrows():
        fn = str(row["filename"]).strip()
        if not fn:
            rel_per_row.append(None)
            continue
        p = _resolve_audio_path(data_root, fn, args.path_prefix)
        rel_per_row.append(p.relative_to(data_root.resolve()).as_posix())

    uniq = sorted({r for r in rel_per_row if r is not None})
    rel_to_ok: dict[str, tuple[bool, str]] = {}

    def check(rel: str) -> tuple[str, bool, str]:
        p = data_root / rel
        ok, why = validate_audio_file(p, target_sr=int(args.target_sr))
        return rel, ok, why

    workers = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(check, rel) for rel in uniq]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="validate_audio"):
            rel, ok, why = fut.result()
            rel_to_ok[rel] = (ok, why)

    bad_rels = {rel for rel, (ok, _) in rel_to_ok.items() if not ok}
    mask = [(r is not None) and (r not in bad_rels) for r in rel_per_row]
    df_out = df.loc[mask].reset_index(drop=True)
    n_bad = int(len(df) - len(df_out))

    reject_log = args.reject_log or (csv_path.parent / f"{csv_path.stem}_rejected_audio.txt")
    reject_log.write_text("\n".join(sorted(bad_rels)) + "\n", encoding="utf-8")

    out_csv = args.out_csv or csv_path
    if args.out_csv is None:
        bak = csv_path.with_suffix(csv_path.suffix + ".bak")
        bak.write_bytes(csv_path.read_bytes())
        print(f"[prune] backup -> {bak}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[prune] rows {len(df)} -> {len(df_out)} (dropped {n_bad}); wrote {out_csv}")
    print(f"[prune] reject list ({len(bad_rels)} unique paths) -> {reject_log}")


if __name__ == "__main__":
    main()
