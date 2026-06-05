import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CkptRef:
    path: Path
    mtime: float
    epoch: int | None = None


def parse_args():
    p = argparse.ArgumentParser(description="Create model soup checkpoint from last-k .ckpt files")
    p.add_argument("--folder", type=Path, required=True, help="folder containing .ckpt files (or a run folder)")
    p.add_argument("--k", type=int, default=5, help="number of checkpoints to soup")
    p.add_argument("--method", type=str, default="avg", choices=("avg", "slerp"), help="soup method")
    p.add_argument("--pattern", type=str, default="*.ckpt", help="glob pattern for checkpoints inside folder")
    p.add_argument(
        "--exclude-last",
        action="store_true",
        help="ignore last.ckpt when selecting checkpoints",
    )
    p.add_argument(
        "--select-by",
        type=str,
        default="epoch",
        choices=("epoch", "mtime"),
        help="how to select last-k checkpoints inside folder",
    )
    p.add_argument(
        "--weights",
        type=str,
        default=None,
        help="optional comma-separated weights, length must equal k (e.g. '1,1,2')",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional output path; default: <folder>_soup_k<k>_<method>.ckpt",
    )
    return p.parse_args()


_EPOCH_RE = re.compile(r"epoch=(\d+)")


def try_parse_epoch(p: Path) -> int | None:
    m = _EPOCH_RE.search(p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _load_ckpt(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt
    raise ValueError(f"Unexpected checkpoint format (no state_dict): {path}")


def _extract_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    sd = ckpt["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("checkpoint['state_dict'] is not a dict")
    out: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if torch.is_tensor(v):
            out[str(k)] = v
    return out


def _validate_same_keys(sds: list[dict[str, torch.Tensor]]) -> list[str]:
    keys0 = set(sds[0].keys())
    for i, sd in enumerate(sds[1:], start=1):
        if set(sd.keys()) != keys0:
            missing = sorted(keys0 - set(sd.keys()))[:30]
            extra = sorted(set(sd.keys()) - keys0)[:30]
            raise ValueError(f"state_dict keys mismatch at idx={i}: missing[:30]={missing} extra[:30]={extra}")
    return sorted(keys0)


def _avg_tensors(tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    # average in float32 for numerical stability, then cast back
    dtype0 = tensors[0].dtype
    if dtype0.is_floating_point:
        acc = torch.zeros_like(tensors[0], dtype=torch.float32)
        wsum = 0.0
        for t, w in zip(tensors, weights, strict=False):
            acc.add_(t.to(torch.float32), alpha=float(w))
            wsum += float(w)
        acc.div_(max(wsum, 1e-12))
        return acc.to(dtype0)
    # non-float tensors: just take from the first checkpoint (should match)
    return tensors[0]


def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    # slerp for floating tensors; falls back to lerp if angle is tiny
    if not (a.dtype.is_floating_point and b.dtype.is_floating_point):
        return a
    aa = a.to(torch.float32).reshape(-1)
    bb = b.to(torch.float32).reshape(-1)
    aa_n = aa / (aa.norm() + 1e-12)
    bb_n = bb / (bb.norm() + 1e-12)
    dot = torch.clamp((aa_n * bb_n).sum(), -1.0, 1.0)
    omega = torch.acos(dot)
    if float(omega) < 1e-6:
        out = (1.0 - t) * a.to(torch.float32) + t * b.to(torch.float32)
        return out.to(a.dtype)
    so = torch.sin(omega)
    w1 = torch.sin((1.0 - t) * omega) / so
    w2 = torch.sin(t * omega) / so
    out = w1 * a.to(torch.float32) + w2 * b.to(torch.float32)
    return out.to(a.dtype)


def main():
    args = parse_args()
    folder = args.folder.resolve()
    if not folder.is_dir():
        raise SystemExit(f"--folder must be a directory: {folder}")
    k = int(args.k)
    if k <= 0:
        raise SystemExit("--k must be > 0")

    ckpts: list[CkptRef] = []
    for p in sorted(folder.glob(str(args.pattern))):
        if p.is_file():
            if args.exclude_last and p.name == "last.ckpt":
                continue
            ckpts.append(CkptRef(path=p, mtime=p.stat().st_mtime, epoch=try_parse_epoch(p)))
    if not ckpts:
        raise SystemExit(f"No checkpoints found under {folder} with pattern={args.pattern!r}")

    select_by = str(args.select_by).strip().lower()
    if select_by == "epoch":
        with_epoch = [c for c in ckpts if c.epoch is not None]
        if len(with_epoch) < k:
            raise SystemExit(
                f"select_by='epoch' requires >=k checkpoints with 'epoch=XX' in filename. "
                f"Found {len(with_epoch)} with epoch out of {len(ckpts)} total."
            )
        with_epoch.sort(key=lambda x: int(x.epoch), reverse=True)
        chosen = with_epoch[:k]
    else:
        ckpts.sort(key=lambda x: x.mtime, reverse=True)
        chosen = ckpts[:k]
    if len(chosen) < k:
        raise SystemExit(f"Found only {len(chosen)} checkpoints, but k={k}")

    weights = None
    if args.weights:
        w = [float(x.strip()) for x in str(args.weights).split(",") if x.strip()]
        if len(w) != k:
            raise SystemExit(f"--weights length must equal k={k}, got {len(w)}")
        weights = w
    else:
        weights = [1.0] * k

    method = str(args.method).strip().lower()
    if method == "slerp" and k != 2:
        raise SystemExit("method=slerp currently supports only k=2")

    ckpt_objs = [_load_ckpt(r.path) for r in chosen]
    sds = [_extract_state_dict(c) for c in ckpt_objs]
    keys = _validate_same_keys(sds)

    out_sd: dict[str, torch.Tensor] = {}
    if method == "avg":
        for key in keys:
            ts = [sd[key] for sd in sds]
            out_sd[key] = _avg_tensors(ts, weights)
    elif method == "slerp":
        t = float(weights[1]) / max(float(weights[0]) + float(weights[1]), 1e-12)
        sd0, sd1 = sds
        for key in keys:
            out_sd[key] = _slerp(sd0[key], sd1[key], t=t)
    else:
        raise SystemExit(f"Unsupported method={method!r}")

    # preserve checkpoint structure from the newest checkpoint
    out_ckpt = dict(ckpt_objs[0])
    out_ckpt["state_dict"] = out_sd
    out_ckpt["soup"] = {
        "method": method,
        "k": k,
        "weights": [float(x) for x in weights],
        "chosen": [str(r.path) for r in chosen],
    }

    out_path = args.out
    if out_path is None:
        out_path = folder / f"{folder.name}_soup_k{k}_{method}.ckpt"
    out_path = out_path.resolve()
    torch.save(out_ckpt, out_path)
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps(out_ckpt["soup"], indent=2), encoding="utf-8")
    print(f"[soup] wrote: {out_path}")
    print(f"[soup] meta:  {meta_path}")


if __name__ == "__main__":
    main()
