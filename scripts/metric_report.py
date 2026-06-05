import argparse
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FOLD_METRICS_RE = re.compile(r"^(?P<exp>.+)_fold(?P<fold>\d+)_metrics\.json$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--oof-dir", type=Path, default=ROOT / "oofs")
    p.add_argument("--exp", type=str, required=True, help="experiment name, e.g. start_point")
    args = p.parse_args()

    oof_dir = args.oof_dir.resolve()

    rows = []
    for path in sorted(oof_dir.glob(f"{args.exp}_fold*_metrics.json")):
        m = FOLD_METRICS_RE.match(path.name)
        if not m or m.group("exp") != args.exp:
            continue
        fold_id = int(m.group("fold"))
        payload = json.loads(path.read_text())
        rows.append((fold_id, payload))

    rows.sort(key=lambda x: x[0])

    print(f"experiment: {args.exp}")
    print(f"oof_dir: {oof_dir}\n")

    aucs = []
    for fold_id, payload in rows:
        print(f"fold {fold_id}: {payload}")
        v = payload.get("macro_auc")
        if v is not None:
            aucs.append(float(v))

    mean = statistics.mean(aucs)
    std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    print()
    print(f"mean macro_auc: {mean}")
    print(f"std macro_auc:  {std}")


if __name__ == "__main__":
    main()
