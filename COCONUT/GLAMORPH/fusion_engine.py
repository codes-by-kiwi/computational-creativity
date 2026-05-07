
import os
import json
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

try:
    from diffusion_generator import generate_fused_image, DiffusionConfig
    _HAS_DIFFUSION = True
except Exception as e:
    _HAS_DIFFUSION = False
    print("❌ Diffusion import failed:", repr(e))


@dataclass
class Item:
    item_id: str
    image_path: str
    category: str
    style_tags: List[str]
    brand: Optional[str] = None
    color: Optional[List[str]] = None
    pattern: Optional[str] = None
    material: Optional[List[str]] = None
    uri: Optional[str] = None
    garment_type: Optional[str] = None      # top/bottom/one_piece
    image_kind: Optional[str] = None        # packshot/model


def load_items(path: str) -> List[Item]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw = data["items"] if isinstance(data, dict) and "items" in data else data

    items: List[Item] = []
    for r in raw:
        items.append(
            Item(
                item_id=str(r["item_id"]),
                image_path=str(r["image_path"]),
                category=str(r.get("category", "")),
                style_tags=list(r.get("style_tags", [])) if r.get("style_tags") else [],
                brand=r.get("brand"),
                color=r.get("color"),
                pattern=r.get("pattern"),
                material=r.get("material"),
                uri=r.get("uri"),
                garment_type=r.get("garment_type"),
                image_kind=r.get("image_kind"),
            )
        )
    return items


def _read_runs(path: str) -> Dict[str, Any]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "runs" in obj:
            return obj
    return {"runs": []}


def append_run(path: str, record: Dict[str, Any]) -> None:
    obj = _read_runs(path)
    obj.setdefault("runs", []).append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def list_categories(items: List[Item]) -> List[str]:
    return sorted({it.category for it in items if it.category})


def list_styles(items: List[Item]) -> List[str]:
    return sorted({s for it in items for s in (it.style_tags or [])})


class ResNet50Embedder(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # 2048-d
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.device = device
        self.to(device)

        w = models.ResNet50_Weights.DEFAULT
        self.pre = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=w.transforms().mean, std=w.transforms().std),
            ]
        )

    @torch.no_grad()
    def embed_path(self, image_path: str) -> Optional[torch.Tensor]:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return None
        x = self.pre(img).unsqueeze(0).to(self.device)
        z = self.backbone(x).squeeze()
        z = z / (z.norm(p=2) + 1e-12)
        return z.cpu()


def build_or_load_embeddings(items: List[Item], cache_path: str = "embeddings.pt") -> Dict[str, torch.Tensor]:
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu")
        return {k: v for k, v in data.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = ResNet50Embedder(device=device)

    out: Dict[str, torch.Tensor] = {}
    for it in items:
        z = emb.embed_path(it.image_path)
        if z is not None:
            out[it.item_id] = z

    if not out:
        raise RuntimeError("No embeddings created. Check image paths.")

    torch.save(out, cache_path)
    return out


def filter_items(
    items: List[Item],
    category: Optional[str] = None,
    must_have_style: Optional[str] = None,
    garment_type: Optional[str] = None,
) -> List[Item]:
    out = []
    for it in items:
        if category and it.category != category:
            continue
        if garment_type and (it.garment_type or "") != garment_type:
            continue
        if must_have_style and must_have_style not in (it.style_tags or []):
            continue
        out.append(it)
    return out


def pick_parents(
    items: List[Item],
    categoryA: Optional[str],
    categoryB: Optional[str],
    styleA: Optional[str],
    styleB: Optional[str],
    garment_typeA: Optional[str] = None,
    garment_typeB: Optional[str] = None,
) -> Tuple[Item, Item]:
    """
    Updated: supports cross-category by allowing categoryA != categoryB (or None).
    """
    A = filter_items(items, category=categoryA, must_have_style=styleA, garment_type=garment_typeA)
    B = filter_items(items, category=categoryB, must_have_style=styleB, garment_type=garment_typeB)

    if not A:
        raise ValueError(f"No candidates for A (category={categoryA}, style={styleA}, garment_type={garment_typeA})")
    if not B:
        raise ValueError(f"No candidates for B (category={categoryB}, style={styleB}, garment_type={garment_typeB})")

    a = random.choice(A)
    b = random.choice(B)
    tries = 0
    while b.item_id == a.item_id and tries < 25:
        b = random.choice(B)
        tries += 1
    return a, b


def fuse_embeddings(zA: torch.Tensor, zB: torch.Tensor, alpha: float) -> torch.Tensor:
    z = alpha * zA + (1.0 - alpha) * zB
    return z / (z.norm(p=2) + 1e-12)


def nearest_neighbor_excluding(
    z: torch.Tensor,
    embeddings: Dict[str, torch.Tensor],
    restrict_ids: List[str],
    exclude_ids: Optional[List[str]] = None,
) -> Tuple[str, float]:
    exclude = set(exclude_ids or [])
    best_id, best_dist = None, float("inf")
    for item_id in restrict_ids:
        if item_id in exclude:
            continue
        v = embeddings[item_id]
        dist = float(1.0 - torch.dot(z, v).clamp(-1, 1).item())
        if dist < best_dist:
            best_dist, best_id = dist, item_id
    if best_id is None:
        raise RuntimeError("Nearest neighbor search failed (empty candidates after exclusion).")
    return best_id, best_dist


def _style_phrase(style_tags: List[str]) -> str:
    tags = [t for t in (style_tags or []) if isinstance(t, str)]
    if not tags:
        return "minimal"
    return ", ".join(sorted(set(tags))[:3])


def generate_one(
    items: List[Item],
    embeddings: Dict[str, torch.Tensor],
    fusion_runs_path: str,
    # Keep "category" for backward compatibility + logging
    category: str,
    styleA: str,
    styleB: str,
    alpha: float,
    mode: str = "retrieval",   # retrieval | diffusion | inpaint_mutation | inpaint_crossover
    output_dir: str = "outputs",
    p_mutation: float = 0.25,
    diffusion_cfg: Optional["DiffusionConfig"] = None,

    # ✅ NEW: force exact parents selected from UI
    parentA_id: Optional[str] = None,
    parentB_id: Optional[str] = None,


    # ✅ NEW: if provided, parents can come from different categories
    parentA_category: Optional[str] = None,
    parentB_category: Optional[str] = None,

    # ✅ Optional: force garment types (useful for dress+jeans composition)
    parentA_garment_type: Optional[str] = None,
    parentB_garment_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Updated: you can now do cross-category fusion by setting:
      parentA_category="dress", parentB_category="trousers"
    or garment type constraints:
      parentA_garment_type="one_piece", parentB_garment_type="bottom"
    """
    alpha = max(0.0, min(1.0, float(alpha)))
    os.makedirs(output_dir, exist_ok=True)

    # Default behavior: same category (backward compatible)
    catA = parentA_category if parentA_category is not None else category
    catB = parentB_category if parentB_category is not None else category

    # a, b = pick_parents(
    #     items,
    #     categoryA=catA,
    #     categoryB=catB,
    #     styleA=styleA,
    #     styleB=styleB,
    #     garment_typeA=parentA_garment_type,
    #     garment_typeB=parentB_garment_type,
    # )

        # --- build quick index ---
    id2item = {it.item_id: it for it in items}

    # ✅ If UI forces exact parents, bypass retrieval/pick logic
    if parentA_id is not None and parentB_id is not None:
        if parentA_id not in id2item or parentB_id not in id2item:
            raise ValueError(f"Forced parent id not found. A={parentA_id}, B={parentB_id}")
        a = id2item[parentA_id]
        b = id2item[parentB_id]
    else:
        # Default behavior: auto-pick parents using constraints
        a, b = pick_parents(
            items,
            categoryA=catA,
            categoryB=catB,
            styleA=styleA,
            styleB=styleB,
            garment_typeA=parentA_garment_type,
            garment_typeB=parentB_garment_type,
        )

    run_id = f"run_{int(time.time())}_{random.randint(0,9999):04d}"
    record: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parents": [a.item_id, b.item_id],
        "constraints": {
            "category": category,
            "parentA_category": catA,
            "parentB_category": catB,
            "styleA": styleA,
            "styleB": styleB,
            "parentA_garment_type": parentA_garment_type,
            "parentB_garment_type": parentB_garment_type,
        },
        "mixing": {"alpha": alpha, "mode": mode},
        "output": {},
        "feedback": [],
    }

    if mode == "retrieval":
        if a.item_id not in embeddings or b.item_id not in embeddings:
            raise ValueError("A parent missing from embeddings cache; rebuild embeddings.")

        zMix = fuse_embeddings(embeddings[a.item_id], embeddings[b.item_id], alpha)

        # restrict by the *logged* category (old behavior). If you want cross-cat retrieval, pass category="any".
        if category == "any":
            restrict = [it.item_id for it in items if it.item_id in embeddings]
        else:
            restrict = [it.item_id for it in items if it.category == category and it.item_id in embeddings]

        out_id, dist = nearest_neighbor_excluding(
            zMix, embeddings, restrict_ids=restrict, exclude_ids=[a.item_id, b.item_id]
        )
        record["output"] = {"retrieved_item_id": out_id, "distance": dist, "image_path": None}
        append_run(fusion_runs_path, record)
        return record

    if mode in ["diffusion"]:
        if not _HAS_DIFFUSION:
            raise RuntimeError(
                "Diffusion requested but diffusion_generator/diffusers not available. "
                "Install: pip install diffusers transformers accelerate safetensors"
            )
        if diffusion_cfg is None:
            diffusion_cfg = DiffusionConfig()

    if mode == "diffusion":
        out_path = os.path.join(output_dir, f"{run_id}_diffusion.png")
        img_path = generate_fused_image(
            image_path_a=a.image_path,
            image_path_b=b.image_path,
            out_path=out_path,
            alpha=alpha,
            cfg=diffusion_cfg,
        )
        record["output"] = {"image_path": img_path, "stage": "fused_img2img"}
        append_run(fusion_runs_path, record)
        return record

    # if mode == "inpaint_mutation":
    #     region = random.choices(["bottom", "top", "center"], weights=[0.45, 0.35, 0.20])[0]
    #     mutation_prompts = {
    #         "bottom": "Keep the upper part unchanged. Redesign the lower part with a new hem and silhouette, add pleats or drape, fashion product photo.",
    #         "top": "Keep the lower part unchanged. Redesign the neckline and sleeves with a new cut, fashion product photo.",
    #         "center": "Keep the garment unchanged except the waist area. Add a new waist detail such as a belt, wrap, or structured seam, fashion product photo.",
    #     }

    #     prompt = f"{diffusion_cfg.prompt}. {mutation_prompts[region]}"
    #     out_path = os.path.join(output_dir, f"{run_id}_mut_{region}.png")
    #     img_path = inpaint_edit(
    #         base_image_path_or_pil=a.image_path,
    #         out_path=out_path,
    #         prompt=prompt,
    #         negative_prompt=diffusion_cfg.negative_prompt,
    #         region=region,
    #         mask_image_path_or_pil=None,
    #         cfg=diffusion_cfg,
    #         strength=max(diffusion_cfg.strength, 0.82),
    #     )

    #     record["output"] = {"image_path": img_path, "stage": "mutated_inpaint", "region": region}
    #     append_run(fusion_runs_path, record)
    #     return record

    # if mode == "inpaint_crossover":
    #     fused_path = os.path.join(output_dir, f"{run_id}_stage1_fused.png")
    #     fused_img = generate_fused_image(
    #         image_path_a=a.image_path,
    #         image_path_b=b.image_path,
    #         out_path=fused_path,
    #         alpha=alpha,
    #         cfg=diffusion_cfg,
    #     )

    #     region = random.choices(["bottom", "top", "center"], weights=[0.55, 0.30, 0.15])[0]
    #     b_style = _style_phrase(b.style_tags)

    #     # Make cross-category hybrids more explicit in the prompt:
    #     cat_hint = f"(Parent A category: {a.category}, Parent B category: {b.category})"
    #     crossover_prompt = (
    #         f"{diffusion_cfg.prompt}. {cat_hint} "
    #         f"Inject design details inspired by the second parent style ({b_style}). "
    #     )

    #     if region == "bottom":
    #         crossover_prompt += "Redesign the lower part: hemline, drape, silhouette, fabric feel, and construction details."
    #     elif region == "top":
    #         crossover_prompt += "Redesign the upper part: sleeves, neckline, collar, and closure."
    #     else:
    #         crossover_prompt += "Redesign the center: waist, belt, wrap detail, and seams."

    #     out_path = os.path.join(output_dir, f"{run_id}_stage2_cross_{region}.png")
    #     img_path = inpaint_edit(
    #         base_image_path_or_pil=fused_img,
    #         out_path=out_path,
    #         prompt=crossover_prompt,
    #         negative_prompt=diffusion_cfg.negative_prompt,
    #         region=region,
    #         mask_image_path_or_pil=None,
    #         cfg=diffusion_cfg,
    #         strength=max(diffusion_cfg.strength, 0.80),
    #     )

    #     record["output"] = {
    #         "image_path": img_path,
    #         "stage": "fused_then_inpaint_crossover",
    #         "region": region,
    #         "intermediate_fused": fused_path,
    #     }
    #     append_run(fusion_runs_path, record)
    #     return record

    # raise ValueError("mode must be one of: retrieval, diffusion, inpaint_mutation, inpaint_crossover")
