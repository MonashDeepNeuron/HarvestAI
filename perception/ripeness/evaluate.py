"""

usage:
  python perception/ripeness/evaluate.py --split test \
      --engine colour-rule perception/models/ripeness/ripeness_v1.pt
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripeness import CLASSES, HARVESTABLE, crop_instance, get_engine  # noqa: E402

CROP_RE = re.compile(r"(?P<split>\w+)/\w+/(?P<sid>\w+)_(?P<num>\d+)_i(?P<iid>\d+)\.jpg")


def load_split(crops_root: Path, strawdi_root: Path, split: str):
    """Rebuild every crop in a split from its source image and instance mask."""
    with open(crops_root / "manifest.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]

    crops, labels, skipped, cache = [], [], 0, {}
    for r in rows:
        m = CROP_RE.match(r["path"])
        if not m:
            skipped += 1
            continue
        num, iid = m["num"], int(m["iid"])
        if num not in cache:   # one entry at a time; crops of a photo are adjacent
            cache = {num: (
                np.asarray(Image.open(strawdi_root / split / "img" / f"{num}.png").convert("RGB")),
                np.asarray(Image.open(strawdi_root / split / "label" / f"{num}.png")))}
        img, lab = cache[num]
        bc = crop_instance(img, lab == iid)
        if bc is None:
            skipped += 1
            continue
        crops.append(bc)
        labels.append(r["label"])
    return crops, labels, skipped


def report(y_true: list[str], y_pred: list[str], title: str) -> None:
    hit = np.array([t == p for t, p in zip(y_true, y_pred)], dtype=float)
    rng = np.random.default_rng(0)
    boot = hit[rng.integers(0, len(hit), size=(5000, len(hit)))].mean(axis=1)
    order = {c: i for i, c in enumerate(CLASSES)}
    mae = np.mean([abs(order[t] - order[p]) for t, p in zip(y_true, y_pred)])

    print(f"\n=== {title} ===")
    print(f"n = {len(y_true)}")
    print(f"accuracy    : {hit.mean():.4f}   95% CI "
          f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]")
    print(f"ordinal MAE : {mae:.4f}")

    print(f"\nper class:\n  {'class':9} {'recall':>8} {'prec':>8} {'f1':>8} {'n':>6}")
    f1s = []
    for c in CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c == p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        rec = tp / (tp + fn) if tp + fn else 0.0
        pre = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0.0
        f1s.append(f1)
        print(f"  {c:9} {rec:8.3f} {pre:8.3f} {f1:8.3f} {tp + fn:6d}")
    print(f"  {'macro-F1':9} {np.mean(f1s):8.3f}")

    print("\nconfusion (rows = truth, cols = predicted):")
    print(f"  {'':9}" + "".join(f"{c:>9}" for c in CLASSES))
    for t in CLASSES:
        print(f"  {t:9}" + "".join(
            f"{sum(1 for a, b in zip(y_true, y_pred) if a == t and b == p):>9}"
            for p in CLASSES))

    ht = [t in HARVESTABLE for t in y_true]
    hp = [p in HARVESTABLE for p in y_pred]
    tp = sum(1 for a, b in zip(ht, hp) if a and b)
    fp = sum(1 for a, b in zip(ht, hp) if not a and b)
    fn = sum(1 for a, b in zip(ht, hp) if a and not b)
    print("\nharvest decision (ripe = pick, turning/unripe = leave):")
    print(f"  precision {tp / (tp + fp) if tp + fp else 0:.3f}  "
          f"recall {tp / (tp + fn) if tp + fn else 0:.3f}  "
          f"(picked-unripe {fp}, missed-ripe {fn})")


def main(a):
    crops, labels, skipped = load_split(Path(a.crops), Path(a.strawdi), a.split)
    print(f"loaded {len(crops)} crops from split={a.split}"
          + (f" ({skipped} skipped)" if skipped else ""))
    for key in a.engine:
        eng = get_engine(key)
        report(labels, [r.label for r in eng.predict(crops)],
               f"{eng.name}  [split={a.split}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", default="data/crops 4")
    p.add_argument("--strawdi", default="data/StrawDI_Db1")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--engine", nargs="+", default=["colour-rule"],
                   help="'colour-rule' and/or paths to .pt checkpoints")
    main(p.parse_args())
