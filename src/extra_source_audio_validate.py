from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.audio_validate import VALIDATE_AUDIO_DEFAULT_WORKERS, validate_audio_file
from src.extra_sources import _normalize_ds_cfg, iter_extra_source_audio_paths


def validate_extra_source_audio_files(
    cfg: dict,
    *,
    out_rel: str = "old_extra_data/extra_sources_rejected_audio.txt",
    workers: int = VALIDATE_AUDIO_DEFAULT_WORKERS,
    target_sr: int = 32000,
) -> dict[str, Any]:
    data_root = Path(cfg["data_root"]).resolve()
    ds_cfg = _normalize_ds_cfg(cfg)
    paths = iter_extra_source_audio_paths(data_root, ds_cfg)
    if not paths:
        return {"n_paths": 0, "n_bad": 0, "out_rel": out_rel, "skipped": True}

    def check(ap: Path) -> tuple[str, bool, str]:
        ok, why = validate_audio_file(ap, target_sr=target_sr)
        rel = ap.resolve().relative_to(data_root).as_posix()
        return rel, ok, why

    bad: set[str] = set()
    workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(check, ap) for ap in paths]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="validate_extra_sources"):
            rel, ok, _why = fut.result()
            if not ok:
                bad.add(rel)

    out_path = data_root / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(bad)) + "\n", encoding="utf-8")

    return {
        "n_paths": len(paths),
        "n_bad": len(bad),
        "out_rel": out_rel,
        "out_path": str(out_path),
        "skipped": False,
    }
