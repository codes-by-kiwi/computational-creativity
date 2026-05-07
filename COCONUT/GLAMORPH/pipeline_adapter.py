from pathlib import Path
from PIL import Image
import json
import random
from fusion_engine import load_items, build_or_load_embeddings, generate_one
from diffusion_generator import DiffusionConfig

import inspect
import traceback
# ----- adjust these local paths -----
DATA_DIR = Path("data")
ITEMS_PATH = DATA_DIR / "items.json"
EMB_PATH = DATA_DIR / "embeddings.pt"
FUSION_RUNS_PATH = DATA_DIR / "fusion_runs.json"
OUTPUT_DIR = Path("outputs")

_ITEMS = None
_EMB = None
_CFG = None

def init_once():
    global _ITEMS, _EMB, _CFG
    if _ITEMS is None:
        _ITEMS = load_items(str(ITEMS_PATH))
    if _EMB is None:
        _EMB = build_or_load_embeddings(_ITEMS, cache_path=str(EMB_PATH))
    if _CFG is None:
        sig = inspect.signature(DiffusionConfig.__init__)
        kwargs = dict(
            model_id="runwayml/stable-diffusion-v1-5",
            prompt="fashion garment, studio lighting, product photo, clean background",
            steps=30,
            strength=0.75,
            guidance_scale=7.5,
            seed=42,
        )
        # Only pass inpaint_model_id if supported by this DiffusionConfig version
        if "inpaint_model_id" in sig.parameters:
            kwargs["inpaint_model_id"] = "runwayml/stable-diffusion-inpainting"

        _CFG = DiffusionConfig(**kwargs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _ITEMS, _EMB, _CFG


DATA_DIR = Path("data")
ITEMS_PATH = DATA_DIR / "items.json"

def _load_items_raw():
    data = json.load(open(ITEMS_PATH, "r", encoding="utf-8"))
    return data["items"] if isinstance(data, dict) and "items" in data else data

def sample_candidates(k=4):
    """Randomly sample k items from dataset."""
    data = json.load(open(ITEMS_PATH, "r", encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data

    sampled = random.sample(items, k)
    out = []
    for it in sampled:
        out.append({
            "item_id": it["item_id"],
            "image_path": it["image_path"]
        })
    return out

def list_candidates(category=None, style=None, limit=60):
    """
    Return a list of (label, item_id, image_path) for UI gallery.
    """
    items = _load_items_raw()
    out = []
    for it in items:
        if category and it.get("category") != category:
            continue
        if style:
            tags = it.get("style_tags") or []
            if style not in tags:
                continue
        out.append((it.get("item_id"), it.get("item_id"), it.get("image_path")))
        if len(out) >= limit:
            break
    return out


def run_generate_with_parents(parentA_id, parentB_id, alpha, mode):
    items, emb, cfg = init_once()
    mode = (mode or "diffusion").strip()

    try:
        print("DEBUG mode repr:", repr(mode))

        rec = generate_one(
            items=items,
            embeddings=emb,
            fusion_runs_path=str(FUSION_RUNS_PATH),
            category="any",      # just for logging
            styleA="manual",
            styleB="manual",
            alpha=float(alpha),
            mode=mode,
            output_dir=str(OUTPUT_DIR),
            diffusion_cfg=cfg,
            parentA_id=parentA_id,
            parentB_id=parentB_id,
        )
    except Exception:
        tb = traceback.format_exc()
        print("=== generate_one ERROR ===")
        print(tb)
        return None, tb, None

    # Defensive checks
    if rec is None:
        msg = "generate_one returned None (unexpected). Check fusion_engine mode branches / returns."
        print("=== generate_one returned None ===")
        return None, msg, None

    if "output" not in rec or rec["output"] is None or "image_path" not in rec["output"]:
        msg = f"generate_one returned record without output.image_path: {rec}"
        print("=== Bad record ===")
        print(msg)
        return None, msg, rec

    out_path = rec["output"]["image_path"]
    out_img = Image.open(out_path).convert("RGB")
    log = f"Mode={mode}\nA={parentA_id}\nB={parentB_id}\nOutput={out_path}"
    return out_img, log, rec

def run_generate(category, styleA, styleB, alpha, mode):
    items, emb, cfg = init_once()
    rec = generate_one(
        items=items,
        embeddings=emb,
        fusion_runs_path=str(FUSION_RUNS_PATH),
        category=category,
        styleA=styleA,
        styleB=styleB,
        alpha=float(alpha),
        mode=mode,
        output_dir=str(OUTPUT_DIR),
        diffusion_cfg=cfg,
    )
    out_path = rec["output"]["image_path"]
    img = Image.open(out_path).convert("RGB")
    log = f"Mode: {mode}\nParents: {rec.get('parents')}\nOutput: {out_path}"
    return img, log
