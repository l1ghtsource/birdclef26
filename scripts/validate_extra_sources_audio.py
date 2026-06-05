import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    from src.utils import cfg_to_dict

    path = path.resolve()
    spec = importlib.util.spec_from_file_location("_v_extra_cfg_", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return cfg_to_dict(mod.cfg)


def main() -> None:
    from src.audio_validate import VALIDATE_AUDIO_DEFAULT_WORKERS
    from src.extra_source_audio_validate import validate_extra_source_audio_files

    ap = argparse.ArgumentParser(description="Validate xc/inat/tsa downloaded audio; write rejected paths list")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--out-rel",
        type=str,
        default="old_extra_data/extra_sources_rejected_audio.txt",
        help="path relative to data_root",
    )
    ap.add_argument("--workers", type=int, default=VALIDATE_AUDIO_DEFAULT_WORKERS)
    ap.add_argument("--target-sr", type=int, default=32000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        cfg["data_root"] = str((ROOT / data_root).resolve())
    st = validate_extra_source_audio_files(
        cfg,
        out_rel=args.out_rel,
        workers=int(args.workers),
        target_sr=int(args.target_sr),
    )
    print(st)


if __name__ == "__main__":
    main()
