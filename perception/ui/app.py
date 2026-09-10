"""Basic Gradio UI to detect strawberries with the trained YOLO model.

Launch:
    python perception/ui/app.py

Then open the printed local URL (default http://127.0.0.1:7860).
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from detector import DEFAULT_MODEL_PATH, get_detector

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - env hint only
    raise SystemExit(
        "gradio is not installed. Run `pip install -r requirements.txt` "
        "inside the harvestai environment, then re-run this script."
    ) from exc


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
    if image is None:
        return None, "Upload an image or take a webcam snapshot to start."
    detector = get_detector(model_path or None)
    result = detector.detect(image, conf=conf)
    confs = [d.confidence for d in result.detections]
    return result.image, _summary(result.count, confs)


def build_ui(model_path: str) -> "gr.Blocks":
    with gr.Blocks(title="HarvestAI · Strawberry Detector") as demo:
        gr.Markdown(
            "# 🍓 Strawberry Detector\n"
            "Upload a photo or use your webcam. The initial YOLO model "
            "(`strawberry_v1.pt`) draws a box + mask around every strawberry it finds."
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(
                    label="Input", sources=["upload", "webcam", "clipboard"], type="numpy"
                )
                conf = gr.Slider(
                    0.05, 0.95, value=0.25, step=0.05, label="Confidence threshold"
                )
                run_btn = gr.Button("Detect strawberries", variant="primary")
            with gr.Column():
                image_out = gr.Image(label="Detections")
                summary = gr.Markdown()

        model_state = gr.State(model_path)
        run_btn.click(run, [image_in, conf, model_state], [image_out, summary])
        image_in.change(run, [image_in, conf, model_state], [image_out, summary])
        conf.release(run, [image_in, conf, model_state], [image_out, summary])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("STRAWBERRY_MODEL", str(DEFAULT_MODEL_PATH)),
        help="Path to the YOLO .pt weights (default: perception/models/strawberry_v1.pt)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    build_ui(args.model).launch(
        server_name=args.host, server_port=args.port, share=args.share
    )


if __name__ == "__main__":
    main()
