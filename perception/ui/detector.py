"""Thin wrapper around the trained YOLO strawberry model.

Keeps all the ultralytics-specific handling in one place so the UI (and any
other caller) just gets an annotated image plus a list of detections back.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ripeness import CLASSES, HARVESTABLE, crop_box_only, crop_instance  # noqa: E402

# perception/ui/detector.py -> perception/models/
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "strawberry_v2.pt"


def list_models() -> list[Path]:
    """Every .pt checkpoint sitting in perception/models/, newest name last.

    Sorted so strawberry_v1, strawberry_v2, ... come out in order. The UI
    defaults to DEFAULT_MODEL_PATH when it's present, otherwise the last one.
    """
    if not MODELS_DIR.is_dir():
        return []
    return sorted(MODELS_DIR.glob("*.pt"))


@dataclass
class Detection:
    confidence: float
    # xyxy pixel coordinates in the source image
    x1: float
    y1: float
    x2: float
    y2: float
    mask: np.ndarray | None = None
    ripeness: str | None = None
    ripeness_conf: float | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def harvestable(self) -> bool:
        return self.ripeness in HARVESTABLE


@dataclass
class DetectionResult:
    image: np.ndarray          # RGB, with boxes/masks drawn on
    detections: list[Detection]

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def has_ripeness(self) -> bool:
        return any(d.ripeness is not None for d in self.detections)

    @property
    def harvestable(self) -> int:
        return sum(1 for d in self.detections if d.harvestable)

    def counts(self) -> dict[str, int]:
        """Per-class tally, in ripeness order, including zeros."""
        return {c: sum(1 for d in self.detections if d.ripeness == c) for c in CLASSES}


# Hardcoded colours for unripe, tunring and ripe strawberries
RIPENESS_RGB = {
    "unripe": (80, 200, 80),
    "turning": (255, 170, 40),
    "ripe": (230, 40, 60),
}
_FALLBACK_RGB = (60, 200, 220)


def draw(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Annotate per-berry ripeness
    """
    import cv2

    out = image.copy()
    s = max(1.0, max(out.shape[:2]) / 640)
    font_scale = 0.55 * s
    text_thick = max(1, round(1.4 * s))
    pad = max(3, round(4 * s))

    for d in detections:
        rgb = RIPENESS_RGB.get(d.ripeness, _FALLBACK_RGB)
        if d.mask is not None:
            tint = np.zeros_like(out)
            tint[d.mask] = rgb
            out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), rgb,
                      max(2, round((3 if d.harvestable else 2) * s)))

    H, W = out.shape[:2]
    placed: list[tuple[int, int, int, int]] = []

    for d in detections:
        rgb = RIPENESS_RGB.get(d.ripeness, _FALLBACK_RGB)
        if d.ripeness:
            tag = f"{d.ripeness} {d.ripeness_conf:.0%}" if d.ripeness_conf else d.ripeness
            if d.harvestable:
                tag += " PICK"
        else:
            tag = f"{d.confidence:.0%}"

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
        bw, bh = tw + 2 * pad, th + 2 * pad

        tx = min(max(0, int(d.x1)), max(0, W - bw))
        y1, y2 = int(d.y1), int(d.y2)

        free = None
        for cy in (y1 - bh, y1, y2 - bh, y2):
            ty = max(0, min(cy, H - bh))
            if not any(tx < px + pw and tx + bw > px and ty < py + ph and ty + bh > py
                       for px, py, pw, ph in placed):
                free = ty
                break
        ty = free if free is not None else max(0, min(y1 - bh, H - bh))
        placed.append((tx, ty, bw, bh))

        cv2.rectangle(out, (tx, ty), (tx + bw, ty + bh), rgb, -1)
        cv2.putText(out, tag, (tx + pad, ty + th + pad - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    text_thick, cv2.LINE_AA)
    return out


class StrawberryDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {self.model_path}. "
                "Expected the trained checkpoint from the Setup notebooks."
            )
        self._model = None

    @property
    def model(self):
        """Load the model lazily so importing this module stays cheap."""
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  
                raise ImportError(
                    "ultralytics is not installed. Run `pip install -r requirements.txt` "
                    "inside the harvestai environment."
                ) from exc
            self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45,
               ripeness_engine=None) -> DetectionResult:
        """Run detection on a single RGB image.
        """
        if image is None:
            raise ValueError("No image provided")

        # ultralytics expects BGR when given a raw numpy array
        bgr = image[:, :, ::-1]
        results = self.model.predict(source=bgr, conf=conf, iou=iou, verbose=False)
        result = results[0]

        masks = self._masks(result, image.shape[:2])

        detections: list[Detection] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for i, ((x1, y1, x2, y2), c) in enumerate(zip(xyxy, confs)):
                detections.append(
                    Detection(float(c), float(x1), float(y1), float(x2), float(y2),
                              mask=masks[i] if masks is not None and i < len(masks) else None)
                )

        detections.sort(key=lambda d: d.confidence, reverse=True)

        if ripeness_engine is not None and detections:
            self._classify(image, detections, ripeness_engine)
            annotated_rgb = draw(image, detections)
        else:
            # result.plot() returns a BGR image with boxes + masks already drawn
            annotated_rgb = result.plot()[:, :, ::-1].copy()

        return DetectionResult(image=annotated_rgb, detections=detections)

    @staticmethod
    def _masks(result, shape: tuple[int, int]) -> list[np.ndarray] | None:
        """
        """
        if getattr(result, "masks", None) is None:
            return None
        import cv2

        h, w = shape
        out = []
        for poly in result.masks.xy:
            m = np.zeros((h, w), dtype=np.uint8)
            if len(poly) >= 3:
                cv2.fillPoly(m, [np.asarray(poly, dtype=np.int32)], 1)
            out.append(m.astype(bool))
        return out

    @staticmethod
    def _classify(image: np.ndarray, detections: list[Detection], engine) -> None:
        """
        """
        crops = []
        for d in detections:
            bc = crop_instance(image, d.mask) if d.mask is not None else None
            if bc is None:
                bc = crop_box_only(image, (d.x1, d.y1, d.x2, d.y2))
            crops.append(bc)

        for d, r in zip(detections, engine.predict(crops)):
            d.ripeness = r.label
            d.ripeness_conf = r.confidence


_DETECTOR_CACHE: dict[str, StrawberryDetector] = {}


def get_detector(model_path: str | Path | None = None) -> StrawberryDetector:
    """Return a detector for the given weights, reusing loaded ones.

    Keeps one StrawberryDetector per checkpoint path so switching models in the
    UI doesn't re-read weights you've already loaded this session.
    """
    key = str(Path(model_path).resolve()) if model_path else str(DEFAULT_MODEL_PATH)
    if key not in _DETECTOR_CACHE:
        _DETECTOR_CACHE[key] = StrawberryDetector(key)
    return _DETECTOR_CACHE[key]
