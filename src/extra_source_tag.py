from pathlib import Path


def infer_extra_source_tag(data_path: Path, meta_path: Path) -> str:
    n = (data_path.name + " " + meta_path.name).lower()
    if "tsa" in n:
        return "tsa"
    if "inat" in n:
        return "inat"
    if "xc" in n:
        return "xc"
    return f"extra_{data_path.name}"
