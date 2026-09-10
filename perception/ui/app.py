"""Basic Gradio UI to detect strawberries with the trained YOLO model(s).

Launch:
    python perception/ui/app.py

Then open the printed local URL (default http://127.0.0.1:7860).

Two tabs:
  * Live   - (default) streams webcam frames through the model. The "Live FPS"
             slider caps how many frames actually get run, so you can match the
             rate to CPU vs GPU (too fast and the stream backs up / goes blank).
  * Image  - upload / webcam snapshot / clipboard, full per-berry breakdown.

The model picker and confidence sit in one row up top; the whole thing is meant
to fit in a single viewport. The picker lists every checkpoint in
perception/models/ and re-scans whenever you switch tabs, so a new
strawberry_v2.pt shows up without a restart.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from detector import DEFAULT_MODEL_PATH, list_models, get_detector

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - env hint only
    raise SystemExit(
        "gradio is not installed. Run `pip install -r requirements.txt` "
        "inside the harvestai environment, then re-run this script."
    ) from exc


def _model_choices() -> list[tuple[str, str]]:
    """(label, value) pairs for the dropdown: label is the filename, value the path."""
    return [(p.name, str(p)) for p in list_models()]


def _default_model() -> str | None:
    """Prefer detector.DEFAULT_MODEL_PATH; fall back to the last checkpoint."""
    values = [v for _, v in _model_choices()]
    if not values:
        return None
    default = str(DEFAULT_MODEL_PATH)
    return default if default in values else values[-1]


def _summary(count: int, confs: list[float]) -> str:
    if count == 0:
        return "### No strawberries detected\nTry lowering the confidence threshold."
    avg = sum(confs) / len(confs)
    lines = [
        f"### {count} strawberr{'y' if count == 1 else 'ies'} detected",
        f"Average confidence: **{avg:.0%}**  |  Best: **{max(confs):.0%}**  |  Weakest: **{min(confs):.0%}**",
        "",
        "| # | Confidence |",
        "| - | ---------- |",
    ]
    lines += [f"| {i} | {c:.1%} |" for i, c in enumerate(confs, start=1)]
    return "\n".join(lines)


def run(image: np.ndarray, conf: float, model_path: str):
    """Image tab: full breakdown."""
    if image is None:
        return None, "Upload an image or take a webcam snapshot to start."
    if not model_path:
        return None, "No model selected. Put a .pt file in perception/models/ and refresh."
    detector = get_detector(model_path)
    result = detector.detect(image, conf=conf)
    confs = [d.confidence for d in result.detections]
    return result.image, _summary(result.count, confs)


# Wall-clock time of the last frame we actually ran inference on. The webcam
# pushes frames faster than a CPU can process them, so we throttle here to the
# FPS the user picked and gr.skip() everything in between — that keeps the
# stream from backing up (which is what makes the Live view freeze / go blank).
_last_infer_t = 0.0


def run_stream(frame: np.ndarray, conf: float, model_path: str, fps: float):
    """Live tab: called once per webcam frame, throttled to `fps`."""
    global _last_infer_t
    if frame is None or not model_path:
        return frame, "Waiting for camera…"

    if time.monotonic() - _last_infer_t < 1.0 / max(fps, 0.5):
        return gr.skip(), gr.skip()

    # Shrink big webcam frames before inference — YOLO letterboxes to 640 anyway,
    # and a smaller annotated frame is a smaller payload back to the browser,
    # which keeps the stream responsive.
    h, w = frame.shape[:2]
    if max(h, w) > 640:
        s = 640 / max(h, w)
        frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    try:
        result = get_detector(model_path).detect(frame, conf=conf)
    except Exception as exc:  # surface it instead of a silent blank frame
        return frame, f"⚠️ {type(exc).__name__}: {exc}"
    _last_infer_t = time.monotonic()

    n = result.count
    return result.image, f"### {n} strawberr{'y' if n == 1 else 'ies'} in frame"


def rescan_models(current: str | None = None):
    """Re-read perception/models/ and keep the current pick if it's still there.

    Wired to each tab's select event, so dropping a new .pt in and clicking
    between tabs makes it appear — no restart, no button.
    """
    choices = _model_choices()
    values = [v for _, v in choices]
    value = current if current in values else _default_model()
    return gr.update(choices=choices, value=value)


def on_model_change(model_path: str, image: np.ndarray, conf: float):
    """Load the newly picked model straight away (so Live doesn't stutter on the
    first frame) and re-run the Image tab on whatever photo is loaded."""
    if not model_path:
        return gr.skip(), gr.skip()
    try:
        get_detector(model_path)  # eager load; cached for subsequent calls
    except Exception as exc:
        gr.Warning(f"{type(exc).__name__}: {exc}")
        return gr.skip(), gr.skip()
    gr.Info(f"Loaded {Path(model_path).name}")
    if image is None:
        return gr.skip(), gr.skip()
    return run(image, conf, model_path)


# Fill the screen without scrolling: use most of the window width, and size the
# image panes to whatever vertical space is left after the controls (clamped so
# they stay sane on very short or very tall windows).
_CSS = """
.gradio-container { max-width: 1400px !important; padding-top: 6px !important; }
.gradio-container h3 { margin: 2px 0 8px; }
#controls { align-items: end; gap: 12px; }
#controls > * { min-width: 0 !important; }
.pane { height: clamp(220px, calc(100vh - 320px), 680px) !important; }
.pane .image-container, .pane [data-testid="image"] { height: 100% !important; }
.pane img { height: 100% !important; object-fit: contain; }
footer { display: none !important; }
"""


def build_ui() -> "gr.Blocks":
    with gr.Blocks(title="HarvestAI · Strawberry Detector") as demo:
        gr.Markdown("### 🍓 Strawberry Detector")

        with gr.Row(elem_id="controls"):
            model_dd = gr.Dropdown(
                label="Model", choices=_model_choices(), value=_default_model(),
                scale=2, min_width=130, container=False,
            )
            conf = gr.Slider(
                0.05, 0.95, value=0.25, step=0.05, label="Confidence", scale=4
            )

        with gr.Tabs():
            with gr.Tab("Live") as live_tab:
                fps = gr.Slider(
                    1, 30, value=4, step=1,
                    label="Live FPS  —  low for CPU, high for GPU",
                )
                with gr.Row(equal_height=True):
                    live_in = gr.Image(
                        label="Webcam", sources=["webcam"], streaming=True,
                        type="numpy", elem_classes="pane",
                    )
                    live_out = gr.Image(
                        label="Detections", interactive=False, elem_classes="pane",
                    )
                live_count = gr.Markdown()

                live_in.stream(
                    run_stream,
                    [live_in, conf, model_dd, fps],
                    [live_out, live_count],
                    stream_every=0.05,
                    concurrency_limit=1,
                    show_progress="hidden",
                )

            with gr.Tab("Image") as image_tab:
                with gr.Row(equal_height=True):
                    image_in = gr.Image(
                        label="Input", sources=["upload", "webcam", "clipboard"],
                        type="numpy", elem_classes="pane",
                    )
                    image_out = gr.Image(
                        label="Detections", interactive=False, elem_classes="pane",
                    )
                with gr.Row():
                    run_btn = gr.Button("Detect strawberries", variant="primary")
                summary = gr.Markdown()

                run_btn.click(run, [image_in, conf, model_dd], [image_out, summary])
                image_in.change(run, [image_in, conf, model_dd], [image_out, summary])
                conf.release(run, [image_in, conf, model_dd], [image_out, summary])

        model_dd.change(
            on_model_change, [model_dd, image_in, conf], [image_out, summary]
        )

        live_tab.select(rescan_models, model_dd, model_dd)
        image_tab.select(rescan_models, model_dd, model_dd)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    if not list_models():
        print(
            f"WARNING: no .pt checkpoints in {DEFAULT_MODEL_PATH.parent} — "
            "the UI will start but has nothing to run."
        )

    build_ui().launch(
        server_name=args.host, server_port=args.port, share=args.share, css=_CSS
    )


if __name__ == "__main__":
    main()
