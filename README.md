# SERNet Cityscapes — Flask Demo with Overlay

> Semantic segmentation (**8 classes**) tailored for **urban street scenes** (Cityscapes-style).  
> Model served from **Hugging Face Hub** with local caching. Flask app includes **upload/URL input** and an **interactive overlay** slider.

[![HF Model](https://img.shields.io/badge/HF%20Hub-SERNet--cityscapes-blue)](https://huggingface.co/juliengatineau/SERNet_cityscapes_trained)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED)](#docker)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Live Demo – Gradio](https://img.shields.io/badge/Demo-Gradio-FF4B4B?logo=gradio)](https://huggingface.co/spaces/juliengatineau/segmentation_SERNet_Cityscapes_Demo)
[![HF Spaces](https://img.shields.io/badge/Spaces-Live%20Demo-black?logo=huggingface)](https://huggingface.co/spaces/juliengatineau/segmentation_SERNet_Cityscapes_Demo)

> **Live demo:** Try the model in your browser via **Gradio** (CPU; first request may be slower due to model warmup).
>
> • Gradio: https://huggingface.co/spaces/juliengatineau/segmentation_SERNet_Cityscapes_Demo
> • HF Spaces: https://huggingface.co/spaces/juliengatineau/segmentation_SERNet_Cityscapes_Demo

---

## TL;DR

-   **Goal:** clean, minimal demo of Cityscapes-style semantic segmentation (8 classes) + interactive **overlay** (mask ↔ image).
-   **Best results:** use **Cityscapes-like** inputs (dashcam street scenes, front-facing, ~daytime).  
    You’ll still get outputs on arbitrary photos, but quality may drop (domain shift).
-   **Included samples:** a few **royalty‑free Pexels** images in `static/images/source/` for convenience. They are **not** Cityscapes; use your own urban images for optimal results.
-   **Model:** `sernet_model.pt` (TorchScript) fetched from HF Hub (auto-cache). You can also bake it into the Docker image.
-   **Prod:** `gunicorn` + multi-stage Docker (slim).

![frankfurt_000000_000294_leftImg8bit](https://github.com/user-attachments/assets/aa82e1e9-325e-4726-afd5-e503f6407da1)

---

## Model

-   HF Hub repo: **`juliengatineau/SERNet_cityscapes_trained`**
-   Weights: **`sernet_model.pt`** (TorchScript; robust loading without custom class code)
-   Metrics on Cityscapes (val): **mIoU ≈ 73%**, **wmIoU ≈ 85%**
-   8 classes (IDs): `void, flat, construction, object, nature, sky, human, vehicle`

> Preprocessing uses torchvision’s `DeepLabV3_ResNet101_Weights.DEFAULT.transforms()` (ImageNet normalization).

---

## Stack

-   Python 3.12, **Flask**
-   **PyTorch** + torchvision (preprocess & tensor pipeline)
-   **Pillow**, **NumPy**, **huggingface-hub**
-   **Gunicorn** (production server), **Docker** (multi-stage, slim runtime)

---

## Project Layout

```
.
├─ app.py                  # Flask app (routes, inference, overlay)
├─ static/
│  ├─ css/styles.css
│  ├─ js/scripts.js
│  └─ images/
│     ├─ source/           # sample images (Pexels) – replace with your own urban scenes
│     ├─ uploads/          # user uploads or downloaded URLs
│     └─ pred/             # predicted masks (PNG)
├─ templates/
│  ├─ index.html           # gallery + upload + URL form
│  └─ display.html         # results + overlay slider
├─ model/
│  └─ sernet_model.pt      # optional (baked into the image or placed locally)
├─ pyproject.toml
├─ poetry.lock             # optional
├─ Dockerfile
└─ README.md
```

---

## Local Setup (Poetry)

> Skip to **Docker** if you prefer containers.

```bash
# Python 3.12 recommended
poetry env use python3.12
poetry install --no-root

# Optional: override defaults
export MODEL_REPO_ID="juliengatineau/SERNet_cityscapes_trained"
export MODEL_FILENAME="sernet_model.pt"

# Dev server
poetry run python app.py
# Or production-like
poetry run gunicorn -w 1 -k gthread -b 0.0.0.0:8000 app:app
```

Open http://127.0.0.1:8000

---

## Docker

### Build & Run

```bash
# Build
docker build -t segmentation-sernet .

# Run
docker run --rm -p 8000:8000 segmentation-sernet
# → http://localhost:8000
```

**What the Dockerfile does**

-   Creates a **virtualenv** in the builder stage (Poetry → export → pip install).
-   Copies the venv + your app into a slim runtime image.
-   Optionally **bakes** the model into `/app/model/sernet_model.pt` for a faster first request.

---

## Configuration (env)

| Variable         | Default                                    | Purpose           |
| ---------------- | ------------------------------------------ | ----------------- |
| `MODEL_REPO_ID`  | `juliengatineau/SERNet_cityscapes_trained` | HF Hub repository |
| `MODEL_FILENAME` | `sernet_model.pt`                          | Weights filename  |
| `PORT`           | `8000`                                     | Serve port        |

> The app loads a local `/app/model/sernet_model.pt` if present, otherwise pulls from HF Hub (cached).

---

## Usage

1. **Index:** pick an image from the gallery, **upload** a file, or paste an **external URL** (http/https).
2. Click **Segment**.
3. **Results:** view **Input**, **Predicted Mask**, and **Overlay**. Move the opacity **slider** to visualize the overlay.

---

## Performance Notes

-   **Best quality**: Cityscapes-like inputs (forward-facing car camera, urban streets, daytime).
-   Resize in-app is tuned for latency/quality balance. Upsampling uses **nearest** to keep label edges crisp.
-   Expect degraded quality on non-urban or unusual viewpoints (**domain shift**).

---

## Troubleshooting

-   **Slow first request**: model load + TorchScript warmup. If you baked the model in Docker, it’s faster.
-   **Overlay looks misaligned**: ensure you didn’t manually alter aspect ratio before running the app.
-   **Artifacts on non‑urban images**: expected due to domain shift; try better matched inputs.

---

## License & Credits

-   Code: **MIT**.
-   Model: SERNet (TorchScript) — **mIoU ≈ 73%**, **wmIoU ≈ 85%** on Cityscapes val.
-   **Cityscapes** dataset: Cordts et al., CVPR 2016 (not included; license restricted).
-   Sample images in `static/images/source/` are **royalty‑free Pexels** photos (for demo only). Replace with your own images if needed.
-   Preprocessing: torchvision / DeepLabV3_ResNet101_Weights.DEFAULT.
