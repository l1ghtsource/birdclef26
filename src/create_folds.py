from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from src.dataset import (
    BirdClefTrainingDataset,
    build_soundscape_items,
    build_soundscape_pseudo_items,
    expand_train_indices_for_upsampling,
    train_audio_csv_specs,
)
from src.extra_sources import build_extra_source_items
from src.taxonomy_merge import build_merged_label_to_idx
from src.utils import norm_class_label


def stratify_label_for_row(primary_labels):
    return primary_labels[0] if len(primary_labels) == 1 else "|".join(sorted(primary_labels))


def n_train_audio_rows(data_root, ds_cfg):
    n = 0
    for csv_rel, _ in train_audio_csv_specs(ds_cfg):
        n += len(pd.read_csv(data_root / csv_rel))
    return n


def collect_stratify_labels(data_root, ds_cfg, label_to_idx, *, soundscape_items=None):
    out = []
    for csv_rel, _ in train_audio_csv_specs(ds_cfg):
        p = data_root / csv_rel
        df = pd.read_csv(p)
        for _, row in df.iterrows():
            out.append(norm_class_label(row["primary_label"]))
    if ds_cfg["use_train_soundscapes"]:
        items = soundscape_items
        if items is None:
            items = build_soundscape_items(
                data_root,
                data_root / ds_cfg["soundscapes_labels_csv"],
                ds_cfg["train_soundscapes_subdir"],
                label_to_idx,
                float(ds_cfg["chunk_duration_s"]),
                float(ds_cfg["soundscape_label_bin_s"]),
            )
        for item in items:
            labs = item.primary_labels
            if not labs:
                out.append("__none__")
            else:
                out.append(stratify_label_for_row(labs))
    return out


def collect_groups(data_root, ds_cfg, *, soundscape_items=None):
    out = []
    for csv_rel, _ in train_audio_csv_specs(ds_cfg):
        p = data_root / csv_rel
        df = pd.read_csv(p)
        for i, row in df.iterrows():
            out.append(f"train_audio:{row['filename'] if 'filename' in row else i}")
    if ds_cfg["use_train_soundscapes"]:
        items = soundscape_items or []
        for item in items:
            out.append(f"train_soundscape:{item.path.name}")
    return out


def ensure_stratifyable(y, n_splits):
    cnt = Counter(y)
    mapped = [lab if cnt[lab] >= n_splits else "__rare__" for lab in y]
    c2 = Counter(mapped)
    if c2.get("__rare__", 0) and c2["__rare__"] < n_splits:
        return np.array(["all"] * len(y), dtype=object)
    return np.array(mapped, dtype=object)


def iter_fold_indices(n, y, val_strategy, n_splits, seed, groups=None):
    idx = np.arange(n)
    if val_strategy == "kf":
        if groups is None:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            it = kf.split(idx)
        else:
            kf = GroupKFold(n_splits=n_splits)
            it = kf.split(idx, groups=np.asarray(groups, dtype=object))
    elif val_strategy == "skf":
        y_arr = ensure_stratifyable(y, n_splits)
        if groups is None:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            it = skf.split(np.zeros(n), y_arr)
        else:
            skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            it = skf.split(np.zeros(n), y_arr, groups=np.asarray(groups, dtype=object))
    else:
        raise NotImplementedError
    return {k: (tr.tolist(), va.tolist()) for k, (tr, va) in enumerate(it)}


def kfold_subpool(global_indices, y_full, val_strategy, n_splits, seed, train_rest, groups_full=None):
    m = len(global_indices)
    y_sub = [y_full[i] for i in global_indices]
    groups_sub = [groups_full[i] for i in global_indices] if groups_full is not None else None
    local_folds = iter_fold_indices(m, y_sub, val_strategy, n_splits, seed, groups=groups_sub)
    rest_sorted = sorted(train_rest)
    out = {}
    for k, (tr_loc, va_loc) in local_folds.items():
        tr_from_pool = [global_indices[i] for i in tr_loc]
        va = [global_indices[i] for i in va_loc]
        tr = sorted(tr_from_pool + rest_sorted)
        out[k] = (tr, va)
    return out


def get_fold_indices(cfg: dict):
    data_root = Path(cfg["data_root"])
    ds_cfg = cfg["dataset"]
    label_to_idx = build_merged_label_to_idx(data_root, ds_cfg)
    n_audio = n_train_audio_rows(data_root, ds_cfg)
    soundscape_items = []
    if ds_cfg["use_train_soundscapes"]:
        soundscape_items = build_soundscape_items(
            data_root,
            data_root / ds_cfg["soundscapes_labels_csv"],
            ds_cfg["train_soundscapes_subdir"],
            label_to_idx,
            float(ds_cfg["chunk_duration_s"]),
            float(ds_cfg["soundscape_label_bin_s"]),
        )
    n_sc = len(soundscape_items)
    pseudo_items = []
    if ds_cfg["use_train_soundscapes"]:
        pseudo_items = build_soundscape_pseudo_items(
            data_root,
            ds_cfg.get("pl_path"),
            data_root / ds_cfg["soundscapes_labels_csv"],
            ds_cfg["train_soundscapes_subdir"],
            label_to_idx,
            float(ds_cfg["chunk_duration_s"]),
            float(ds_cfg["soundscape_label_bin_s"]),
            pl_filter=bool(ds_cfg.get("pl_filter", False)),
            pl_filter_thr=float(ds_cfg.get("pl_filter_thr", 0.5)),
            pl_zero_unconf=bool(ds_cfg.get("pl_zero_unconf", False)),
            pl_zero_unconf_thr=float(ds_cfg.get("pl_zero_unconf_thr", 0.1)),
            log_stats=False,
        )
    n_pseudo = len(pseudo_items)
    extra_items, _ = build_extra_source_items(cfg, label_to_idx, log=False)
    n_extra = len(extra_items)
    strat_labels = collect_stratify_labels(data_root, ds_cfg, label_to_idx, soundscape_items=soundscape_items)
    groups = collect_groups(data_root, ds_cfg, soundscape_items=soundscape_items)
    n = len(strat_labels)

    pool = cfg["val_split_pool"]
    if pool not in ("all", "train_audio", "soundscape"):
        raise NotImplementedError

    if cfg["do_full_retrain"]:
        return {"train": [list(range(n))], "val": None}

    n_splits = int(cfg["n_splits"])

    pseudo_start = n + n_extra
    extra_indices = list(range(n, n + n_extra)) if n_extra > 0 else []
    pseudo_indices = list(range(pseudo_start, pseudo_start + n_pseudo))

    if pool == "all" or n_audio == 0 or n_sc == 0:
        fold_to_tv = iter_fold_indices(n, strat_labels, cfg["val_strategy"], n_splits, int(cfg["seed"]), groups=groups)
    elif pool == "train_audio":
        fold_to_tv = kfold_subpool(
            list(range(0, n_audio)),
            strat_labels,
            cfg["val_strategy"],
            n_splits,
            int(cfg["seed"]),
            train_rest=list(range(n_audio, n)),
            groups_full=groups,
        )
    elif pool == "soundscape":
        fold_to_tv = kfold_subpool(
            list(range(n_audio, n)),
            strat_labels,
            cfg["val_strategy"],
            n_splits,
            int(cfg["seed"]),
            train_rest=list(range(0, n_audio)),
            groups_full=groups,
        )
    else:
        raise NotImplementedError

    train_only_extra = set(extra_indices) | set(pseudo_indices)
    if train_only_extra:
        for k, (tr, va) in fold_to_tv.items():
            tr_new = sorted(set(tr) | train_only_extra)
            fold_to_tv[k] = (tr_new, va)

    train_out = []
    val_out = []
    for f in cfg["curr_folds"]:
        tr, va = fold_to_tv[f]
        train_out.append(tr)
        val_out.append(va)

    return {"train": train_out, "val": val_out}


def get_folds(cfg: dict):
    fold_indices = get_fold_indices(cfg)
    cfg_train = {**cfg, "is_train": True}
    cfg_val = {**cfg, "is_train": False}
    base_train = BirdClefTrainingDataset(cfg_train)
    base_val = BirdClefTrainingDataset(cfg_val)

    if cfg["do_full_retrain"]:
        tr = list(range(len(base_train)))
        if cfg["do_upsampling"]:
            tr = expand_train_indices_for_upsampling(tr, base_train.items, int(cfg["upsampling_n"]), int(cfg["seed"]))
        return {"train": [torch.utils.data.Subset(base_train, tr)], "val": None}

    train_datasets = []
    val_datasets = []
    for fold_i, (tr_idx, va_idx) in enumerate(zip(fold_indices["train"], fold_indices["val"], strict=False)):
        if cfg["do_upsampling"]:
            fold_items = [base_train.items[i] for i in tr_idx]
            tr_idx = expand_train_indices_for_upsampling(
                tr_idx,
                fold_items,
                int(cfg["upsampling_n"]),
                int(cfg["seed"]) + fold_i,
            )
        train_datasets.append(torch.utils.data.Subset(base_train, tr_idx))
        val_datasets.append(torch.utils.data.Subset(base_val, va_idx))

    return {"train": train_datasets, "val": val_datasets}
