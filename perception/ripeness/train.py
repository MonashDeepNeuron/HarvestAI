
#usage: python perception/ripeness/train.py [--name ripeness_v2] [--epochs 60]

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripeness import CLASSES, MODELS_DIR, PAD_RGB, build_model, preprocess  # noqa: E402

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def augment(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.7:
        img = img.rotate(random.uniform(-20, 20), resample=Image.BILINEAR,
                         fillcolor=PAD_RGB)
    if random.random() < 0.5:                      # resolution jitter, see above
        w, h = img.size
        s = random.uniform(0.4, 1.0)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
    if random.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))
    return img


class CropDataset(Dataset):
    def __init__(self, root: str | Path, split: str, train: bool = False):
        self.root, self.train = Path(root), train
        with open(self.root / "manifest.csv", newline="") as f:
            self.rows = [r for r in csv.DictReader(f)
                         if r["split"] == split and r["label"] in CLASS_TO_IDX]
        if not self.rows:
            raise ValueError(f"no rows for split={split!r} under {self.root}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["path"]).convert("RGB")
        if self.train:
            img = augment(img)
        return preprocess(img), CLASS_TO_IDX[r["label"]]

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount([CLASS_TO_IDX[r["label"]] for r in self.rows],
                             minlength=len(CLASSES)).astype(np.float64)
        w = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
        return torch.tensor(w / w.mean(), dtype=torch.float32)


def _worker_init(worker_id: int) -> None:
    """Fork hands every worker the same RNG state; without this the augmentation
    stream repeats identically across workers each epoch."""
    s = torch.initial_seed() % 2**31
    random.seed(s + worker_id)
    np.random.seed((s + worker_id) % 2**31)


@torch.no_grad()
def macro_f1(model, loader, device) -> tuple[float, float]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps += model(x.to(device)).argmax(1).cpu().tolist()
        ys += y.tolist()

    f1s = []
    for c in range(len(CLASSES)):
        tp = sum(1 for a, b in zip(ys, ps) if a == c == b)
        fp = sum(1 for a, b in zip(ys, ps) if a != c and b == c)
        fn = sum(1 for a, b in zip(ys, ps) if a == c and b != c)
        pre = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pre * rec / (pre + rec) if pre + rec else 0.0)
    return float(np.mean([a == b for a, b in zip(ys, ps)])), float(np.mean(f1s))


def main(a):
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    tr = CropDataset(a.data, "train", train=True)
    va = CropDataset(a.data, "val")
    print(f"device {device}   train {len(tr)}   val {len(va)}   seed {a.seed}")

    kw = dict(num_workers=4, worker_init_fn=_worker_init, persistent_workers=True)
    tl = DataLoader(tr, batch_size=a.batch, shuffle=True, **kw)
    vl = DataLoader(va, batch_size=a.batch, **kw)

    model = build_model(len(CLASSES), pretrained=True).to(device)
    # Freeze the stem and early blocks first: with 959 crops, fine-tuning the
    # whole backbone from the start just overfits.
    for name, p in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1", "layer2")):
            p.requires_grad = False

    crit = nn.CrossEntropyLoss(weight=tr.class_weights().to(device), label_smoothing=0.05)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, best_state, stale = {"f1": -1.0}, None, 0
    for epoch in range(1, a.epochs + 1):
        if epoch == a.freeze_epochs + 1:
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr * 0.1)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, a.epochs - a.freeze_epochs))
            print(f"  -- unfroze backbone, lr -> {a.lr * 0.1:.2e}")

        model.train()
        t0, total, seen = time.time(), 0.0, 0
        for x, y in tl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            seen += len(y)
        sched.step()

        acc, f1 = macro_f1(model, vl, device)
        flag = ""
        if f1 > best["f1"]:
            best, stale, flag = {"f1": f1, "acc": acc, "epoch": epoch}, 0, "  *"
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"epoch {epoch:3d}/{a.epochs}  loss {total / seen:.4f}  "
              f"val acc {acc:.4f}  macro-F1 {f1:.4f}  ({time.time() - t0:.1f}s){flag}")
        if stale >= a.patience:
            print(f"early stop: no val improvement in {a.patience} epochs")
            break

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / f"{a.name}.pt"
    torch.save({"classes": list(CLASSES), "state_dict": best_state,
                "val": best, "args": vars(a)}, out)
    print(f"\nbest val macro-F1 {best['f1']:.4f} (epoch {best['epoch']})\nsaved -> {out}")
    print(f"\ncompare against the baseline:\n  python perception/ripeness/evaluate.py "
          f"--split test --engine colour-rule {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/crops 4")
    p.add_argument("--name", default="ripeness_v1")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--freeze-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
