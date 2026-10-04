from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ("unripe", "turning", "ripe")

HARVESTABLE = frozenset({"ripe"})

SCAN_PAD, EXPORT_PAD = 0.08, 0.25  
MAX_SIDE = 256                      
MIN_PX = 24                         # skip berries thinner than this
T_LOW, T_HIGH = 0.05, 0.25        
IMG_SIZE = 224
PAD_RGB = (114, 114, 114)          

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "ripeness"
COLOUR_RULE_KEY = "colour-rule"


# --------------------------------------------------------------------- cropping

@dataclass
class BerryCrop:
    image: np.ndarray                 
    berry_pixels: np.ndarray | None    


def redness(px: np.ndarray) -> float:
    """negative means greener than red"""
    if len(px) == 0:
        return 0.0
    return float(np.median((px[:, 0] - px[:, 1]) / (px.sum(1) + 1e-6)))


def _cut(image: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """"""
    h, w = image.shape[:2]
    for pad, bump in ((SCAN_PAD, 1), (EXPORT_PAD, 0)):
        dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
        x0, y0 = max(0, x0 - dx), max(0, y0 - dy)
        x1, y1 = min(w, x1 + dx + bump), min(h, y1 + dy + bump)

    crop = Image.fromarray(image[y0:y1, x0:x1])
    if max(crop.size) > MAX_SIDE:
        sc = MAX_SIDE / max(crop.size)
        crop = crop.resize((max(1, round(crop.width * sc)),
                            max(1, round(crop.height * sc))), Image.LANCZOS)
    return np.asarray(crop)


def crop_instance(image: np.ndarray, mask: np.ndarray) -> BerryCrop | None:
    """"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    if min(y1 - y0, x1 - x0) < MIN_PX:
        return None
    return BerryCrop(_cut(image, x0, y0, x1, y1), image[mask].astype(np.float32))


def crop_box_only(image: np.ndarray, xyxy) -> BerryCrop:
    """Fallback with no mask. The colour rule degrades here and says so."""
    x0, y0, x1, y1 = (int(round(v)) for v in xyxy)
    return BerryCrop(_cut(image, x0, y0, x1, y1), None)


def preprocess(image: np.ndarray):
    """"""
    import torch

    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    w, h = img.size
    sc = IMG_SIZE / max(w, h)
    img = img.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS)
    square = Image.new("RGB", (IMG_SIZE, IMG_SIZE), PAD_RGB)
    square.paste(img, ((IMG_SIZE - img.size[0]) // 2, (IMG_SIZE - img.size[1]) // 2))

    t = torch.from_numpy(np.asarray(square, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (t - mean) / std


def build_model(n_classes: int = 3, pretrained: bool = False):
    """Resnet 18"""
    import torch.nn as nn
    from torchvision import models

    m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.fc.in_features, n_classes))
    return m



@dataclass
class Ripeness:
    label: str
    confidence: float

    @property
    def harvestable(self) -> bool:
        return self.label in HARVESTABLE


def _label_for(r: float) -> str:
    return "unripe" if r < T_LOW else ("turning" if r < T_HIGH else "ripe")


class ColourRuleEngine:
    name = "Colour rule"

    def predict(self, crops: list[BerryCrop]) -> list[Ripeness]:
        out = []
        for c in crops:
            px, degraded = c.berry_pixels, False
            if px is None or len(px) == 0:
                px, degraded = c.image.reshape(-1, 3).astype(np.float32), True

            r = redness(px)
            conf = 0.5 + 0.5 * math.tanh(min(abs(r - T_LOW), abs(r - T_HIGH)) / 0.05)
            out.append(Ripeness(_label_for(r), min(conf, 0.6) if degraded else conf))
        return out


class CnnEngine:
    """Fine tuned resnet"""

    def __init__(self, weights: str | Path):
        self.weights = Path(weights)
        if not self.weights.exists():
            raise FileNotFoundError(
                f"No ripeness checkpoint at {self.weights}. train  with "
                "`python perception/ripeness/train.py`.")
        self.name = self.weights.stem
        self._model = None
        self._classes = CLASSES

    @property
    def model(self):
        if self._model is None:
            import torch
            self._device = torch.device(
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu")
            ckpt = torch.load(self.weights, map_location="cpu", weights_only=False)
            self._classes = tuple(ckpt.get("classes", CLASSES))
            m = build_model(len(self._classes))
            m.load_state_dict(ckpt["state_dict"])
            self._model = m.eval().to(self._device)
        return self._model

    def predict(self, crops: list[BerryCrop]) -> list[Ripeness]:
        if not crops:
            return []
        import torch

        model = self.model  # also sets _device / _classes
        batch = torch.stack([preprocess(c.image) for c in crops]).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(model(batch), dim=1).cpu().numpy()

        return [Ripeness(self._classes[int(p.argmax())], float(p.max())) for p in probs]


_CACHE: dict[str, object] = {}


def list_cnn_models() -> list[Path]:
    """"""
    return sorted(MODELS_DIR.glob("*.pt")) if MODELS_DIR.is_dir() else []


def engine_choices() -> list[tuple[str, str]]:
    """ui dropdown"""
    return ([("Colour rule (baseline)", COLOUR_RULE_KEY)]
            + [(p.stem, str(p)) for p in list_cnn_models()])


def get_engine(key: str | None):
    """Return the engine for a dropdown value, reusing already-loaded ones."""
    key = key or COLOUR_RULE_KEY
    if key not in _CACHE:
        _CACHE[key] = ColourRuleEngine() if key == COLOUR_RULE_KEY else CnnEngine(key)
    return _CACHE[key]
