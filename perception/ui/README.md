# Strawberry Detector UI

A small Gradio app that runs the initial YOLO model (`perception/models/strawberry_v1.pt`)
on an uploaded photo or webcam frame and draws a box + mask around each strawberry.

## Setup

If you have the `harvestai` conda env:

```bash
conda activate harvestai
pip install -r requirements.txt   # from the repo root; adds gradio + pillow
```

No conda? Use a plain venv (works with the system Python 3.14 on this box):

```bash
python3 -m venv .venv                 # from the repo root; .venv is git-ignored
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install ultralytics gradio opencv-python-headless pillow
```

The CPU wheels are used deliberately — this NUC has no NVIDIA GPU, and it keeps
the install ~3 GB smaller. Inference on a 800 px photo takes well under a second.

## Launch

```bash
python perception/ui/app.py          # or .venv/bin/python perception/ui/app.py
```

Open the printed URL (default <http://127.0.0.1:7860>).

### Options

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--model PATH` | `perception/models/strawberry_v1.pt` | Use a different `.pt` checkpoint (e.g. `strawberry_v2.pt`) |
| `--port N` | `7860` | Change the port |
| `--host H` | `127.0.0.1` | Bind address |
| `--share` | off | Create a temporary public Gradio link |

`STRAWBERRY_MODEL` env var is also honoured as the default model path.

## Files

- `app.py` – Gradio interface, launch entry point
- `detector.py` – `StrawberryDetector` wrapper around ultralytics `YOLO`; returns the
  annotated image plus a list of `Detection` boxes. Reusable outside the UI.
