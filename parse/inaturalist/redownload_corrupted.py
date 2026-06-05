import argparse
import csv
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import av


def parse_label_basename(filename_cell: str) -> tuple[str, str]:
    s = (filename_cell or "").strip().replace("\\", "/")
    if not s:
        return "_invalid", "unknown.ogg"
    parts = s.split("/")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return "_flat", parts[0]


def download(url: str, dest: Path, timeout_s: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "birds_hand download_corrupted_audios_to_ogg/1.0"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = r.read()
    dest.write_bytes(data)


def transcode_to_ogg_vorbis(src: Path, dst: Path, sample_rate: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    inp = av.open(str(src), mode="r", options={"analyzeduration": "10M", "probesize": "10M"})
    try:
        astreams = [s for s in inp.streams if s.type == "audio" and s.codec]
        if not astreams:
            raise ValueError("no audio stream in file")
        a_in = astreams[0]
    except Exception:
        inp.close()
        raise

    out = av.open(str(dst), "w", format="ogg")
    a_out = out.add_stream("libvorbis", rate=sample_rate, layout="mono")

    resampler = av.audio.resampler.AudioResampler(
        format="fltp",
        layout="mono",
        rate=sample_rate,
    )

    try:
        for frame in inp.decode(a_in):
            for rframe in resampler.resample(frame):
                for packet in a_out.encode(rframe):
                    out.mux(packet)
        for packet in a_out.encode(None):
            out.mux(packet)
    finally:
        out.close()
        inp.close()


def row_dict(row: list[str], header: list[str]) -> dict[str, Any]:
    return {h: (row[i] if i < len(row) else "") for i, h in enumerate(header)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("data/corrupted_audios.csv"),
        help="Path to corrupted_audios.csv (default: data/corrupted_audios.csv)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Output root (default: ./output)",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=32_000,
        help="Resample to this rate (Hz), mono; matches project train sample_rate (default: 32000)",
    )
    p.add_argument("--timeout", type=int, default=120, help="Per-download timeout seconds")
    p.add_argument("--limit", type=int, default=0, help="Process only first N rows (0 = all)")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if target .ogg already exists and is non-empty",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions only")
    args = p.parse_args()

    if av is None:
        print("error: PyAV is required. Install: pip install av", file=sys.stderr)
        return 1

    if not args.csv.is_file():
        print(f"error: csv not found: {args.csv}", file=sys.stderr)
        return 1

    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        if "url" not in header or "filename" not in header:
            print("error: csv must have columns: filename, url", file=sys.stderr)
            return 1
        rows = list(reader)

    n_ok = 0
    n_skip = 0
    n_fail = 0
    lim = args.limit if args.limit > 0 else len(rows)

    for i, row in enumerate(rows[:lim]):
        d = row_dict(row, header)
        fn = str(d.get("filename", "")).strip()
        url = str(d.get("url", "")).strip()
        if not url or not fn:
            print(f"[{i + 1}] skip: empty url or filename")
            n_skip += 1
            continue

        label, base = parse_label_basename(fn)
        if not base.lower().endswith(".ogg"):
            base = f"{Path(base).stem}.ogg"
        out_ogg: Path = args.output / label / base

        if args.skip_existing and out_ogg.is_file() and out_ogg.stat().st_size > 0:
            n_skip += 1
            continue

        if args.dry_run:
            print(f"would: {url!r} -> {out_ogg}")
            n_ok += 1
            continue

        with tempfile.TemporaryDirectory(prefix="dl_ogg_") as tmpd:
            tmp = Path(tmpd)
            suf = Path(urllib.parse.urlparse(url).path).suffix
            if not suf or len(suf) > 6:
                suf = ".bin"
            raw = tmp / f"in_{i}{suf}"
            try:
                download(url, raw, args.timeout)
            except (OSError, urllib.error.URLError, TimeoutError) as e:
                print(f"[{i + 1}] fail download {fn}: {e}", file=sys.stderr)
                n_fail += 1
                continue
            try:
                transcode_to_ogg_vorbis(raw, out_ogg, args.sample_rate)
            except (OSError, ValueError, RuntimeError, av.FFmpegError) as e:  # type: ignore[union-attr]
                print(f"[{i + 1}] fail encode {fn}: {e}", file=sys.stderr)
                n_fail += 1
                continue
        n_ok += 1
        print(f"[{i + 1}] ok: {out_ogg}")

    print(f"done: ok={n_ok} skip={n_skip} fail={n_fail} output={args.output.resolve()}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
