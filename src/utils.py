import ast
from enum import Enum
from types import SimpleNamespace

import pandas as pd


def parse_list_cell(cell):
    return [str(x).strip() for x in ast.literal_eval(str(cell).strip())]


def norm_class_label(x):
    return str(x).strip() if not pd.isna(x) else ""


def time_hms_to_seconds(hms):
    h, m, s = str(hms).strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def filter_taxonomy_labels(labels, label_to_idx):
    return [lab for lab in labels if lab in label_to_idx]


def cfg_to_dict(obj):
    if obj is None:
        return None
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, SimpleNamespace):
        return {k: cfg_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: cfg_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(cfg_to_dict(v) for v in obj)
    return obj
