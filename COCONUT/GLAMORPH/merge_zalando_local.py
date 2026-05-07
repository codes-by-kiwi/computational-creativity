import json, os, re, random
from pathlib import Path
from collections import Counter

import requests

# ======================
# Paths (LOCAL)
# ======================
DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "zalando_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Put your 4 json filenames here (they are under data/)
# Example: data/a.json, data/b.json, ...
ZALANDO_JSONS = [
    DATA_DIR / "dataset_zalando-item-search-scraper_1300products_pullovers.json",
    DATA_DIR / "dataset_zalando-item-search-scraper_3140products_dresses.json",
    DATA_DIR / "dataset_zalando-product-search-scraper_jeans.json",
    DATA_DIR / "dataset_zalando-product-search-scraper_broeken.json",
]

# ======================
# Sampling targets
# ======================
TARGET_TOTAL = 200
MIN_PER_CATEGORY = {"dress": 40, "pullover": 40, "top": 20, "skirt": 40, "trousers": 40}
MAX_PER_CATEGORY = {"dress": 60, "pullover": 60, "top": 40, "skirt": 60, "trousers": 60}

# ======================
# Utilities (same logic as your script)
# ======================
def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s)

def pick_img_url(p: dict):
    keys = [
        "extra_large_packshot_image", "large_packshot_image", "medium_packshot_image", "small_packshot_image",
        "extra_large_default_media", "large_default_media", "medium_default_media", "small_default_media"
    ]
    for k in keys:
        v = p.get(k)
        if isinstance(v, dict) and v.get("uri"):
            return v["uri"]
    return None

def infer_image_kind(img_url: str) -> str:
    return "packshot" if "filter=packshot" in (img_url or "") else "model"

def guess_style_tags(text: str):
    t = (text or "").lower()
    tags = []
    if any(k in t for k in ["cocktail", "kant", "lace", "chiffon", "satijn", "satin", "ruched", "wikkel", "wrap"]):
        tags.append("formal")
    if any(k in t for k in ["gebreide", "knit", "jersey", "rib", "basic", "eenvoudig", "naadloos"]):
        tags.append("casual")
    if any(k in t for k in ["denim", "oversized", "hood", "street", "sneaker"]):
        tags.append("streetwear")
    if not tags:
        tags.append("minimal")
    return sorted(set(tags))

def infer_category(p: dict) -> str:
    name = (p.get("name") or "").lower()
    uri  = (p.get("uri") or "").lower()
    alt_default = ((p.get("default_media_info") or {}).get("alternative_text") or "").lower()
    alt_packshot = ((p.get("packshot_image_info") or {}).get("alternative_text") or "").lower()

    # 1) URI first
    text_uri = uri
    if any(k in text_uri for k in ["broek", "trousers", "pants", "jeans", "legging"]):
        return "trousers"
    if any(k in text_uri for k in ["rok", "skirt"]):
        return "skirt"
    if any(k in text_uri for k in ["jurk", "dress"]):
        return "dress"
    if any(k in text_uri for k in ["trui", "pullover", "sweater", "hoodie", "cardigan", "sweatshirt"]):
        return "pullover"
    if any(k in text_uri for k in ["blouse", "shirt", "t-shirt", "tee", "top"]):
        return "top"

    # 2) Name next
    text_name = name
    if any(k in text_name for k in ["broek", "trousers", "pants", "jeans", "legging"]):
        return "trousers"
    if any(k in text_name for k in ["rok", "skirt"]):
        return "skirt"
    if any(k in text_name for k in ["jurk", "dress"]):
        return "dress"
    if any(k in text_name for k in ["trui", "pullover", "sweater", "hoodie", "cardigan", "sweatshirt"]):
        return "pullover"
    if any(k in text_name for k in ["blouse", "shirt", "t-shirt", "tee", "top"]):
        return "top"

    # 3) Alt last
    text_alt = " ".join([alt_default, alt_packshot])
    if any(k in text_alt for k in ["broek", "trousers", "pants", "jeans", "legging"]):
        return "trousers"
    if any(k in text_alt for k in ["rok", "skirt"]):
        return "skirt"
    if any(k in text_alt for k in ["jurk", "dress"]):
        return "dress"
    if any(k in text_alt for k in ["trui", "pullover", "sweater", "hoodie", "cardigan", "sweatshirt"]):
        return "pullover"
    if any(k in text_alt for k in ["blouse", "shirt", "t-shirt", "tee", "top"]):
        return "top"

    sil = (p.get("silhouette") or "unknown").lower()
    return sil

def garment_type_from_category(cat: str) -> str:
    cat = (cat or "").lower()
    if cat in ["dress", "jumpsuit", "onepiece", "one_piece"]:
        return "one_piece"
    if cat in ["trousers", "skirt", "shorts"]:
        return "bottom"
    if cat in ["pullover", "top", "blouse", "shirt", "t-shirt", "tee", "jacket", "coat"]:
        return "top"
    return "unknown"

def minimums_met(counts: Counter) -> bool:
    return all(counts.get(k, 0) >= v for k, v in MIN_PER_CATEGORY.items())

def load_json_any(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        return []
    return data

def download(url: str, out_path: Path, timeout=25) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except Exception:
        return False

if __name__ == "__main__":
    print("Checking input files...")
    for p in ZALANDO_JSONS:
        print(f"  {p} -> {p.exists()}")

    merged = []
    for p in ZALANDO_JSONS:
        if not p.exists():
            print("Skipping missing file:", p)
            continue
        merged.extend(load_json_any(p))

    print("\nTotal raw records (merged):", len(merged))

    # De-dupe by id/sku
    seen = set()
    deduped = []
    for rec in merged:
        item_id = rec.get("id") or rec.get("sku")
        if not item_id:
            continue
        if item_id in seen:
            continue
        seen.add(item_id)
        deduped.append(rec)

    print("Deduped records:", len(deduped))

    random.seed(42)
    random.shuffle(deduped)

    items_out = []
    skipped = 0
    counts = Counter()

    for p in deduped:
        if len(items_out) >= TARGET_TOTAL and minimums_met(counts):
            break

        item_id = p.get("id") or p.get("sku")
        if not item_id:
            skipped += 1
            continue

        img_url = pick_img_url(p)
        if not img_url:
            skipped += 1
            continue

        category = infer_category(p).lower()
        if category not in MIN_PER_CATEGORY:
            continue
        if counts[category] >= MAX_PER_CATEGORY[category]:
            continue

        fname = safe_name(str(item_id)) + ".jpg"
        out_path = OUT_DIR / fname
        if not out_path.exists():
            ok = download(img_url, out_path)
            if not ok:
                skipped += 1
                continue

        brand = ((p.get("brand") or {}).get("name")) if isinstance(p.get("brand"), dict) else None
        alt = (p.get("default_media_info") or {}).get("alternative_text", "") or ""

        items_out.append({
            "item_id": item_id,
            "image_path": str(out_path),  # absolute-ish local path
            "category": category,
            "garment_type": garment_type_from_category(category),
            "image_kind": infer_image_kind(img_url),
            "brand": brand,
            "style_tags": guess_style_tags(alt),
            "uri": p.get("uri"),
        })
        counts[category] += 1

    out_json_path = DATA_DIR / "items.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump({"items": items_out}, f, indent=2, ensure_ascii=False)

    print("\nConverted items:", len(items_out))
    print("Skipped (download/no-img/etc):", skipped)
    print("Category counts:", dict(counts))
    print("Minimums met:", minimums_met(counts))
    print("Saved:", out_json_path.resolve())
    if items_out:
        print("Example:", items_out[0])



