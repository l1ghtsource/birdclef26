from pathlib import Path

import pandas as pd

from src.utils import norm_class_label


def _ds_get(ds_cfg, key: str, default=None):
    if isinstance(ds_cfg, dict):
        return ds_cfg.get(key, default)
    return getattr(ds_cfg, key, default)


def collect_extra_folder_labels(data_root: Path, ds_cfg) -> set[str]:
    if not bool(_ds_get(ds_cfg, "extra_merge_taxonomy_from_folders", True)):
        return set()
    out: set[str] = set()
    for rel in _ds_get(ds_cfg, "extra_sources_data") or []:
        root = Path(rel) if Path(rel).is_absolute() else data_root / rel
        if not root.is_dir():
            continue
        for ch in root.iterdir():
            if ch.is_dir() and not ch.name.startswith("."):
                out.add(norm_class_label(ch.name))
    return out


def build_merged_label_to_idx(data_root: Path, ds_cfg) -> dict[str, int]:
    data_root = Path(data_root)
    tax_path = data_root / _ds_get(ds_cfg, "taxonomy_csv")
    tax = pd.read_csv(tax_path)
    base = [norm_class_label(x) for x in tax["primary_label"]]
    merged = sorted(set(base) | collect_extra_folder_labels(data_root, ds_cfg))
    return {lab: i for i, lab in enumerate(merged)}


def merged_num_classes(data_root: Path, ds_cfg) -> int:
    return len(build_merged_label_to_idx(data_root, ds_cfg))
