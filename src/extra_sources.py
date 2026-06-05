import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.dataset import DatasetItem, build_soundscape_items, train_audio_csv_specs
from src.extra_source_tag import infer_extra_source_tag
from src.utils import cfg_to_dict, norm_class_label


def _ds_get(ds_cfg: dict, key: str, default: Any = None) -> Any:
    if isinstance(ds_cfg, dict):
        return ds_cfg.get(key, default)
    return getattr(ds_cfg, key, default)


_DEFAULT_EXTRA_CAP_SOURCE_PRIORITY: tuple[str, ...] = ("inat", "xc", "tsa")


def _parse_extra_cap_source_priority(ds_cfg) -> tuple[str, ...]:
    raw = _ds_get(ds_cfg, "extra_max_class_num_source_priority", None)
    if raw is None:
        return _DEFAULT_EXTRA_CAP_SOURCE_PRIORITY
    if isinstance(raw, (str, bytes)):
        seq = (raw,)
    else:
        seq = raw
    allowed_tags = frozenset(_DEFAULT_EXTRA_CAP_SOURCE_PRIORITY)
    seen: set[str] = set()
    order: list[str] = []
    for x in seq:
        t = str(x).strip().lower()
        if t in allowed_tags and t not in seen:
            seen.add(t)
            order.append(t)
    for t in _DEFAULT_EXTRA_CAP_SOURCE_PRIORITY:
        if t not in seen:
            order.append(t)
    return tuple(order)


def _cap_row_sort_key(row: tuple[str, str, Path, str], tag_priority: tuple[str, ...]) -> tuple[int, str]:
    _src, _prim, path, tag = row
    try:
        pr = tag_priority.index(tag)
    except ValueError:
        pr = len(tag_priority)
    return (pr, str(path))


def parse_manual_geo(s: str) -> tuple[float, float, float, float]:
    m = re.match(
        r"\s*\[\s*([^;\]]+)\s*;\s*([^;\]]+)\s*\]\s*/\s*\[\s*([^;\]]+)\s*;\s*([^;\]]+)\s*\]\s*",
        str(s).strip(),
    )
    if not m:
        raise ValueError(f"extra_filter_geo manual mode expects '[lat_min;lat_max]/[lon_min;lon_max]', got {s!r}")
    lat_min, lat_max, lon_min, lon_max = (float(x.strip()) for x in m.groups())
    return lat_min, lat_max, lon_min, lon_max


def compute_train_geo_bounds(data_root: Path, train_csv: str) -> tuple[float, float, float, float] | None:
    p = data_root / train_csv
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return None
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    m = lat.notna() & lon.notna()
    if not bool(m.any()):
        return None
    return float(lat[m].min()), float(lat[m].max()), float(lon[m].min()), float(lon[m].max())


def resolve_geo_bounds(
    data_root: Path,
    ds_cfg: dict,
    extra_filter_geo: str,
) -> tuple[float, float, float, float] | None:
    g = str(extra_filter_geo).strip().lower()
    if g in ("none", ""):
        return None
    if g == "like_train":
        tc = _ds_get(ds_cfg, "train_csv", "train.csv")
        b = compute_train_geo_bounds(data_root, tc)
        if b is None:
            raise RuntimeError(
                "extra_filter_geo=like_train but could not compute bounds from train.csv (missing lat/lon?)."
            )
        return b
    return parse_manual_geo(extra_filter_geo)


def row_passes_geo(
    lat: float | None,
    lon: float | None,
    bounds: tuple[float, float, float, float] | None,
) -> bool:
    if bounds is None:
        return True
    lat_min, lat_max, lon_min, lon_max = bounds
    if (
        lat is None
        or lon is None
        or (isinstance(lat, float) and np.isnan(lat))
        or (isinstance(lon, float) and np.isnan(lon))
    ):
        return True
    return lat_min <= float(lat) <= lat_max and lon_min <= float(lon) <= lon_max


def count_core_train_per_label(
    data_root: Path,
    ds_cfg: dict,
    label_to_idx: dict[str, int],
) -> Counter[str]:
    cnt: Counter[str] = Counter()
    for csv_rel, _ in train_audio_csv_specs(ds_cfg):
        df = pd.read_csv(data_root / csv_rel)
        for _, r in df.iterrows():
            lab = norm_class_label(r["primary_label"])
            if lab in label_to_idx:
                cnt[lab] += 1
    if _ds_get(ds_cfg, "use_train_soundscapes", False):
        items = build_soundscape_items(
            data_root,
            data_root / _ds_get(ds_cfg, "soundscapes_labels_csv"),
            _ds_get(ds_cfg, "train_soundscapes_subdir"),
            label_to_idx,
            float(_ds_get(ds_cfg, "chunk_duration_s", 5.0)),
            float(_ds_get(ds_cfg, "soundscape_label_bin_s", 5.0)),
        )
        for it in items:
            for lab in it.primary_labels:
                if lab in label_to_idx:
                    cnt[lab] += 1
    return cnt


_AUDIO_EXT = frozenset({".ogg", ".wav", ".mp3", ".m4a", ".flac", ".opus"})


def _sound_id_from_inat_url(url: str) -> str | None:
    m = re.search(r"/sounds/(\d+)", str(url))
    return m.group(1) if m else None


def _iter_audio_files_under_extra_root(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name.startswith("."):
            continue
        if p.suffix.lower() in _AUDIO_EXT:
            out.append(p)
    return out


def iter_extra_source_audio_paths(data_root: Path, ds_cfg: dict) -> list[Path]:
    roots_raw = _ds_get(ds_cfg, "extra_sources_data") or []
    if not roots_raw:
        return []
    data_root = data_root.resolve()
    out: list[Path] = []
    for raw in roots_raw:
        p = Path(raw)
        if not p.is_absolute():
            p = data_root / p
        if p.is_dir():
            out.extend(_iter_audio_files_under_extra_root(p))
    seen: set[str] = set()
    uniq: list[Path] = []
    for ap in out:
        k = str(ap.resolve())
        if k not in seen:
            seen.add(k)
            uniq.append(ap.resolve())
    return uniq


def _label_dir_from_extra_audio(audio: Path, data_root: Path) -> str | None:
    try:
        rel = audio.relative_to(data_root)
    except ValueError:
        return None
    if len(rel.parts) < 2:
        return None
    return str(rel.parts[0])


def _meta_row_keys(row: pd.Series, tag: str) -> list[str]:
    keys: list[str] = []
    if tag == "inat":
        sid = _sound_id_from_inat_url(str(row.get("sound_url", "")))
        if sid:
            keys.extend([sid, f"iNat{sid}"])
        if "id" in row.index and pd.notna(row["id"]):
            try:
                oid = str(int(float(row["id"])))
            except (ValueError, TypeError):
                oid = str(row["id"]).strip()
            if oid:
                keys.append(oid)
    elif tag == "xc":
        fn = row.get("file_name")
        if fn is None or (isinstance(fn, float) and np.isnan(fn)):
            return []
        base = str(fn).strip().splitlines()[0].strip()
        base = Path(base).name
        m = re.search(r"(XC\d+)", base, re.I)
        if m:
            keys.append(m.group(1))
        keys.append(Path(base).stem)
    elif tag == "tsa":
        uid = row.get("unique_identifier")
        if uid is None or str(uid).strip() == "":
            return []
        u = str(uid).strip()
        keys.append(u.replace(":", "_").replace(" ", "_"))
        keys.append(u)
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        k = str(k).strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _audio_lookup_keys(audio: Path, tag: str) -> list[str]:
    stem = audio.stem
    keys: list[str] = [stem, stem.replace(" ", "_")]
    if tag == "inat":
        m = re.search(r"iNat(\d+)", stem, re.I)
        if m:
            keys.extend([m.group(1), f"iNat{m.group(1)}"])
    elif tag == "xc":
        m = re.search(r"(XC\d+)", stem, re.I)
        if m:
            keys.append(m.group(1))
    elif tag == "tsa":
        pass
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        k = str(k).strip()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _build_meta_index(meta_root: Path, tag: str) -> tuple[dict[str, dict[str, pd.Series]], int]:
    by_taxon: dict[str, dict[str, pd.Series]] = {}
    n_rows = 0
    for csv_path in _iter_meta_csvs(meta_root):
        taxon = csv_path.stem
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        n_rows += len(df)
        bucket = by_taxon.setdefault(taxon, {})
        for _, row in df.iterrows():
            for k in _meta_row_keys(row, tag):
                bucket[k] = row
    return by_taxon, n_rows


def _read_lat_lon(row: pd.Series, source: str) -> tuple[float | None, float | None]:
    if source in ("inat", "xc"):
        lat = row["latitude"] if "latitude" in row.index else None
        lon = row["longitude"] if "longitude" in row.index else None
        try:
            lat = float(lat) if lat is not None and str(lat).strip() != "" and pd.notna(lat) else None
        except Exception:
            lat = None
        try:
            lon = float(lon) if lon is not None and str(lon).strip() != "" and pd.notna(lon) else None
        except Exception:
            lon = None
        return lat, lon
    return None, None


def _iter_meta_csvs(meta_root: Path) -> list[Path]:
    if not meta_root.is_dir():
        return []
    return sorted(meta_root.glob("*.csv"))


def _normalize_ds_cfg(cfg: dict) -> dict:
    d = cfg.get("dataset")
    if isinstance(d, dict):
        return d
    if d is None:
        return {}
    return cfg_to_dict(d)


def build_extra_source_items(
    cfg: dict,
    label_to_idx: dict[str, int],
    *,
    log: bool = True,
) -> tuple[list[DatasetItem], dict[str, Any]]:
    data_root = Path(cfg["data_root"])
    ds_cfg = _normalize_ds_cfg(cfg)

    paths_data = _ds_get(ds_cfg, "extra_sources_data") or []
    paths_meta = _ds_get(ds_cfg, "extra_sources_meta") or []
    if not paths_data or not paths_meta:
        return [], {"empty": True}

    if len(paths_data) != len(paths_meta):
        raise ValueError(
            "extra_sources_data and extra_sources_meta must have same length, "
            f"got {len(paths_data)} vs {len(paths_meta)}"
        )

    extra_filter_geo = str(_ds_get(ds_cfg, "extra_filter_geo", "none") or "none")
    bounds = resolve_geo_bounds(data_root, ds_cfg, extra_filter_geo)

    rare_raw = _ds_get(ds_cfg, "extra_rare_thr", "none")
    rare_thr: int | None
    if rare_raw is None or str(rare_raw).lower() == "none":
        rare_thr = None
    else:
        rare_thr = int(rare_raw)

    cap_raw = _ds_get(ds_cfg, "extra_max_class_num", "none")
    if cap_raw is None or str(cap_raw).lower() == "none":
        cap_n = None
    else:
        cap_n = int(cap_raw)

    cap_source_priority = _parse_extra_cap_source_priority(ds_cfg)

    train_counts = count_core_train_per_label(data_root, ds_cfg, label_to_idx)

    # (source_key, primary_label, path, tag) — tag for geo helper only
    all_rows: list[tuple[str, str, Path, str]] = []

    per_source: dict[str, dict[str, Any]] = {}

    for data_p_raw, meta_p_raw in zip(paths_data, paths_meta, strict=True):
        data_root_src = Path(data_p_raw)
        meta_root_src = Path(meta_p_raw)
        if not data_root_src.is_absolute():
            data_root_src = data_root / data_root_src
        if not meta_root_src.is_absolute():
            meta_root_src = data_root / meta_root_src

        tag = infer_extra_source_tag(data_root_src, meta_root_src)
        src_key = f"extra_{tag}"
        meta_by_taxon, meta_row_total = _build_meta_index(meta_root_src, tag)

        st: dict[str, Any] = {
            "audio_files_scanned": 0,
            "meta_rows_indexed": meta_row_total,
            "skipped_unknown_label_folder": 0,
            "skipped_rare_taxon": 0,
            "meta_matched": 0,
            "meta_unmatched": 0,
            "dropped_geo": 0,
            "accepted_pre_cap": 0,
        }

        for audio_path in _iter_audio_files_under_extra_root(data_root_src):
            st["audio_files_scanned"] += 1
            prim = _label_dir_from_extra_audio(audio_path, data_root_src)
            if prim is None or prim not in label_to_idx:
                st["skipped_unknown_label_folder"] += 1
                continue

            if rare_thr is not None and train_counts.get(prim, 0) >= rare_thr:
                st["skipped_rare_taxon"] += 1
                continue

            row: pd.Series | None = None
            bucket = meta_by_taxon.get(prim)
            if bucket:
                for lk in _audio_lookup_keys(audio_path, tag):
                    if lk in bucket:
                        row = bucket[lk]
                        st["meta_matched"] += 1
                        break
            if row is None:
                st["meta_unmatched"] += 1

            # geo only when we have a meta row with usable coordinates (otherwise keep sample)
            if tag in ("inat", "xc") and bounds is not None and row is not None:
                lat, lon = _read_lat_lon(row, tag)
                if lat is not None and lon is not None:
                    if not row_passes_geo(lat, lon, bounds):
                        st["dropped_geo"] += 1
                        continue

            all_rows.append((src_key, prim, audio_path, tag))
            st["accepted_pre_cap"] += 1

        per_source[src_key] = st

    # allowed_extra = max(0, cap_n - train_count). If over quota, keep rows by
    # extra_max_class_num_source_priority (default inat -> xc -> tsa), then path (stable).
    if cap_n is not None and all_rows:
        by_lab: dict[str, list[tuple[str, str, Path, str]]] = {}
        for t in all_rows:
            by_lab.setdefault(t[1], []).append(t)
        capped: list[tuple[str, str, Path, str]] = []
        n_dropped_cap = 0
        for lab, rows in by_lab.items():
            core = int(train_counts.get(lab, 0))
            if core >= cap_n:
                n_dropped_cap += len(rows)
                continue
            allowed_extra = cap_n - core
            if len(rows) <= allowed_extra:
                capped.extend(rows)
            else:
                rows_sorted = sorted(rows, key=lambda r: _cap_row_sort_key(r, cap_source_priority))
                keep = rows_sorted[:allowed_extra]
                n_dropped_cap += len(rows) - allowed_extra
                capped.extend(keep)
        all_rows = capped
    else:
        n_dropped_cap = 0

    items = [
        DatasetItem(
            source=t[0],
            path=t[2],
            primary_labels=(t[1],),
            secondary_labels=(),
        )
        for t in all_rows
    ]

    per_source_final = Counter(it.source for it in items)

    extra_per_label = Counter(t[1] for t in all_rows)
    total_train = sum(train_counts.values())
    total_extra = len(all_rows)
    total_new = total_train + total_extra

    labels_sorted = sorted(label_to_idx.keys(), key=lambda x: label_to_idx[x])

    rows_csv = []
    for lab in labels_sorted:
        tc = int(train_counts.get(lab, 0))
        ec = int(extra_per_label.get(lab, 0))
        nc = tc + ec
        tf = (tc / total_train) if total_train > 0 else 0.0
        ef = (ec / total_extra) if total_extra > 0 else 0.0
        nf = (nc / total_new) if total_new > 0 else 0.0
        rows_csv.append(
            {
                "label": lab,
                "train_freq": tf,
                "train_count": tc,
                "extra_freq": ef,
                "extra_count": ec,
                "new_train_freq": nf,
                "new_train_count": nc,
            }
        )

    stats: dict[str, Any] = {
        "empty": False,
        "extra_filter_geo": extra_filter_geo,
        "geo_bounds": bounds,
        "extra_rare_thr": rare_thr,
        "extra_max_class_num": cap_n,
        "extra_max_class_num_source_priority": list(cap_source_priority),
        "n_extra_total": len(items),
        "n_dropped_cap": n_dropped_cap,
        "per_source": per_source,
        "per_source_final_counts": dict(per_source_final),
        "label_table": rows_csv,
        "train_total": total_train,
        "extra_total": total_extra,
    }

    if log:
        cap_prio_s = f" cap_source_priority={list(cap_source_priority)}" if cap_n is not None else ""
        print(
            "[extra_sources] "
            f"filter_geo={extra_filter_geo!r} bounds={bounds} "
            f"rare_thr={rare_thr} max_total_per_class={cap_n}{cap_prio_s} "
            f"total_extra={len(items)} dropped_cap={n_dropped_cap}"
        )
        for sk, st in per_source.items():
            fin = int(per_source_final.get(sk, 0))
            print(
                f"  [{sk}] audio_scanned={st['audio_files_scanned']} meta_rows_indexed={st['meta_rows_indexed']} "
                f"unknown_label_dir={st['skipped_unknown_label_folder']} skipped_rare_taxon={st['skipped_rare_taxon']} "
                f"meta_matched={st['meta_matched']} meta_unmatched={st['meta_unmatched']} "
                f"geo_dropped={st['dropped_geo']} accepted_pre_cap={st['accepted_pre_cap']} final_after_cap={fin}"
            )

    out_path = _ds_get(ds_cfg, "extra_stats_csv", None)
    if out_path and log:
        outp = Path(out_path)
        if not outp.is_absolute():
            outp = data_root / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_csv).to_csv(outp, index=False)
        stats["stats_csv_path"] = str(outp)
        print(f"[extra_sources] wrote label stats csv -> {outp}")

    return items, stats


def n_extra_items_from_cfg(cfg: dict, label_to_idx: dict[str, int]) -> int:
    items, _ = build_extra_source_items(cfg, label_to_idx, log=False)
    return len(items)
