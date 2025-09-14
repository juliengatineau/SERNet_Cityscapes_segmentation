from __future__ import annotations
import os
from io import BytesIO
from uuid import uuid4
from threading import Lock

import requests
from werkzeug.utils import secure_filename

from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from PIL import Image
import numpy as np
from matplotlib import colors
import torch
from huggingface_hub import hf_hub_download
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import torch.nn.functional as F


# -----------------------------
# Chemins
# -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_IMG = os.path.join(ROOT, "static", "images")
SOURCE_DIR = os.path.join(STATIC_IMG, "source")   # images de démo
UPLOAD_DIR = os.path.join(STATIC_IMG, "uploads")  # images utilisateur (upload / URL)
PRED_DIR   = os.path.join(STATIC_IMG, "pred")     # masques prédits

os.makedirs(SOURCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PRED_DIR,   exist_ok=True)

# -----------------------------
# Modèle
# -----------------------------
MODEL_REPO_ID   = os.getenv("MODEL_REPO_ID", "juliengatineau/SERNet_cityscapes_trained")
MODEL_FILENAME  = os.getenv("MODEL_FILENAME", "sernet_model.pt")
MODEL_LOCAL_PT  = os.path.join(ROOT, "model", "sernet_model.pt")  # (optionnel) baked via Docker

WEIGHTS   = DeepLabV3_ResNet101_Weights.DEFAULT
PREPROCESS = WEIGHTS.transforms()

_MODEL: torch.nn.Module | None = None
_MODEL_LOCK = Lock()

# palette simple (8 classes)
PALETTE = ['b','g','r','c','m','y','k','w']  # 0..7
ID2CAT  = {0:'void',1:'flat',2:'construction',3:'object',4:'nature',5:'sky',6:'human',7:'vehicle'}


# "Qualité" pilote par env (tu peux fixer en dur si tu veux)
TTA_SCALES = tuple(float(s) for s in os.getenv("TTA_SCALES", "1.0").split(","))  # ex "0.75,1.0,1.25"
TTA_FLIP   = os.getenv("TTA_FLIP", "0").lower() in {"1","true","yes"}
SMOOTH_K   = int(os.getenv("SMOOTH_K", "3"))  # 1=off, 3=3x3
TARGET_SHORT = int(os.getenv("TARGET_SHORT", "480"))  # côté court visé
ALIGN32      = 32  # stride réseau


# -----------------------------
# Utils
# -----------------------------
def _list_images(folder: str) -> list[str]:
    """Liste les images (png/jpg/jpeg) triées alpha."""
    exts = {".png", ".jpg", ".jpeg"}
    try:
        files = [f for f in os.listdir(folder) if os.path.splitext(f.lower())[1] in exts]
    except FileNotFoundError:
        return []
    return sorted(files)

def _load_model() -> torch.nn.Module:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        # 1) Priorité au poids local (baked)
        if os.path.exists(MODEL_LOCAL_PT):
            path = MODEL_LOCAL_PT
        else:
            # 2) Essaye le cache HF local (pas de réseau), sinon réseau
            try:
                path = hf_hub_download(MODEL_REPO_ID, MODEL_FILENAME, local_files_only=True)
            except Exception:
                path = hf_hub_download(MODEL_REPO_ID, MODEL_FILENAME)

        # 3) Chargement (TorchScript préférable)
        try:
            m = torch.jit.load(path, map_location="cpu").eval()
        except Exception:
            obj = torch.load(path, map_location="cpu")
            if hasattr(obj, "eval"):
                m = obj.eval()
            else:
                raise TypeError("Poids ni TorchScript ni nn.Module.")
        # 4) Warmup (réduit la latence de la 1re requête)
        try:
            dummy = PREPROCESS(Image.new("RGB", (960, 480))).unsqueeze(0)
            with torch.inference_mode():
                out = m(dummy)
                _ = (out["out"] if isinstance(out, dict) else out).shape
        except Exception:
            pass

        _MODEL = m
        return _MODEL

def _palette_mask(seg: np.ndarray) -> np.ndarray:
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for cid in ID2CAT:
        mask = seg == cid
        r, g, b = colors.to_rgb(PALETTE[cid])
        out[mask, 0] = r
        out[mask, 1] = g
        out[mask, 2] = b
    return (out * 255).astype(np.uint8)

def _fit_resize_and_pad(pil: Image.Image, short_side: int = TARGET_SHORT, align: int = ALIGN32):
    """
    Redimensionne en gardant le ratio pour que le côté court = short_side,
    puis pad à droite/bas pour que (H, W) soient multiples de `align`.
    Retourne (image_padded, meta) pour recadrer ensuite.
    """
    Resample = getattr(Image, "Resampling", Image)
    w0, h0 = pil.size
    scale = short_side / min(w0, h0)
    w1, h1 = int(round(w0 * scale)), int(round(h0 * scale))
    img = pil.resize((w1, h1), Resample.LANCZOS)

    pad_w = (align - (w1 % align)) % align
    pad_h = (align - (h1 % align)) % align
    if pad_w or pad_h:
        canvas = Image.new("RGB", (w1 + pad_w, h1 + pad_h), (0, 0, 0))
        canvas.paste(img, (0, 0))
        img = canvas
    meta = {"orig_size": (w0, h0), "scaled_size": (w1, h1), "pad": (pad_w, pad_h)}
    return img, meta


def _predict_png(raw: bytes) -> bytes:
    """
    Inférence robuste hors-Cityscapes : ratio conservé + padding, TTA optionnelle,
    lissage des logits puis remontée à la taille d'origine.
    """
    model = _load_model()
    pil_orig = Image.open(BytesIO(raw)).convert("RGB")
    img_pad, meta = _fit_resize_and_pad(pil_orig)
    (w0, h0) = meta["orig_size"]
    (w1, h1) = meta["scaled_size"]
    (pad_w, pad_h) = meta["pad"]

    def _forward_on_pil(pil_img: Image.Image):
        x = PREPROCESS(pil_img).unsqueeze(0)
        with torch.inference_mode():
            out = model(x)
        logits = out["out"] if isinstance(out, dict) else out  # (1,C,H,W)
        return logits

    # ---- TTA léger (moyenne des logits) ----
    logits_acc = None
    for s in (TTA_SCALES or (1.0,)):
        # redimensionne l'image paddée
        if s != 1.0:
            new_w, new_h = int(round(img_pad.width * s)), int(round(img_pad.height * s))
            pil_s = img_pad.resize((new_w, new_h), getattr(Image, "Resampling", Image).BILINEAR)
        else:
            pil_s = img_pad

        logit = _forward_on_pil(pil_s)

        # flip horizontal (optionnel)
        if TTA_FLIP:
            xflip = pil_s.transpose(Image.FLIP_LEFT_RIGHT)
            logit_f = _forward_on_pil(xflip)
            logit_f = torch.flip(logit_f, dims=[3])  # débloque le flip
            logit = (logit + logit_f) * 0.5

        # ramène à la taille paddée de base (h_pad, w_pad)
        logit = F.interpolate(logit, size=(img_pad.height, img_pad.width), mode="bilinear", align_corners=False)

        logits_acc = logit if logits_acc is None else (logits_acc + logit)

    logits = logits_acc / float(len(TTA_SCALES) if TTA_SCALES else 1.0)

    # ---- Lissage 3x3 des logits (réduit le bruit) ----
    if SMOOTH_K and SMOOTH_K > 1:
        pad = SMOOTH_K // 2
        logits = F.avg_pool2d(logits, kernel_size=SMOOTH_K, stride=1, padding=pad)

    # ---- Argmax puis recadrage (on enlève le padding), puis resize à la taille d'origine ----
    seg = torch.argmax(logits, 1).cpu().numpy()[0]  # (Hpad, Wpad)
    seg = seg[:h1, :w1]  # enlève le pad bas/droite si présent

    # colorisation puis resize NEAREST à la taille d'entrée initiale (w0,h0)
    color = _palette_mask(seg)  # (h1,w1,3)
    mask_img = Image.fromarray(color).resize((w0, h0), getattr(Image, "Resampling", Image).NEAREST)

    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    return buf.getvalue()

# -------- Ingestion user files (upload/url) --------
def _save_user_image(pil: Image.Image, stem: str) -> str:
    """Sauvegarde une image user (RGB) en PNG dans UPLOAD_DIR. Retourne filename."""
    uid = uuid4().hex[:8]
    filename = f"{stem}_{uid}.png"
    pil.convert("RGB").save(os.path.join(UPLOAD_DIR, filename))
    return filename

def _ingest_upload(file_storage) -> str:
    raw = file_storage.read()
    if not raw:
        abort(400, "Uploaded file is empty")
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        abort(400, "Uploaded file is not a valid image")
    stem = secure_filename(os.path.splitext(file_storage.filename or "upload")[0]) or "upload"
    return _save_user_image(pil, f"user_{stem}")

def _ingest_external_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        abort(400, "Invalid URL")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = r.content
    except Exception as e:
        abort(400, f"Failed to download image: {e}")
    try:
        pil = Image.open(BytesIO(raw)).convert("RGB")
    except Exception:
        abort(400, "Downloaded file is not a valid image")
    base = secure_filename(os.path.splitext(os.path.basename(url.split("?")[0]) or "url_image")[0]) or "url_image"
    return _save_user_image(pil, f"url_{base}")

# -----------------------------
# App
# -----------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

@app.route("/images/<image_type>/<path:filename>")
def serve_image(image_type, filename):
    mapping = {
        "source": SOURCE_DIR,
        "uploads": UPLOAD_DIR,
        "pred": PRED_DIR,
    }
    directory = mapping.get(image_type)
    if not directory:
        abort(404)
    return send_from_directory(directory, filename)

@app.route("/")
def index():
    names = _list_images(SOURCE_DIR)  # liste simple
    return render_template("index.html", image_source_names=names)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Priorité d'entrée :
      - upload_file (File)
      - external_url (str)
      - image_url (str) depuis le dataset local `source/`
    """
    up = request.files.get("upload_file")
    ext_url = (request.form.get("external_url") or "").strip()
    ds_url = (request.form.get("image_url") or "").strip()

    if up and up.filename:
        real_image = _ingest_upload(up)
        storage = "uploads"
    elif ext_url:
        real_image = _ingest_external_url(ext_url)
        storage = "uploads"
    elif ds_url:
        real_image = os.path.basename(ds_url)
        storage = "source"
    else:
        abort(400, "No input provided")

    src_root = SOURCE_DIR if storage == "source" else UPLOAD_DIR
    src_path = os.path.join(src_root, real_image)
    if not os.path.isfile(src_path):
        abort(404, f"Image not found: {real_image}")

    with open(src_path, "rb") as f:
        pred_png = _predict_png(f.read())

    stem = real_image.rsplit("_", 1)[0] if storage == "source" else os.path.splitext(real_image)[0]
    pred_name = f"{stem}_pred.png"
    with open(os.path.join(PRED_DIR, pred_name), "wb") as f:
        f.write(pred_png)

    return redirect(url_for(
        "display",
        image_id=stem,
        image_type=storage,      # 'source' | 'uploads'
        real_image=real_image,
        predicted_mask=pred_name
    ), code=303)

@app.route("/display")
def display():
    return render_template("display.html",
        image_id       = request.args.get("image_id", "NA"),
        image_type     = request.args.get("image_type", "source"),
        real_image     = request.args.get("real_image"),
        predicted_mask = request.args.get("predicted_mask"),
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")), debug=True)
