"""Build the cluster-lab image-processing showcase.

Picks a handful of random successfully-segmented bikes and renders, for each,
the four stages the visual pipeline goes through to turn a raw photo into the
``clip_vector_visual`` that drives visual clustering:

    1. Original photo
    2. YOLOv8 detection      (bounding box drawn on the source)
    3. SAM mask              (segmentation mask overlaid on the source)
    4. White-bg crop         (the actual CLIP input)

Output lands under ``media/showcase/<offer_id>/`` plus a ``manifest.json`` the
dashboard reads. Re-running reshuffles the selection — it wipes the showcase
directory first.

    python ml/build_showcase.py                # 3 random bikes
    python ml/build_showcase.py --count 4
    python ml/build_showcase.py --seed 42      # reproducible pick

Models load lazily (YOLOv8n ~6 MB, SAM2-base ~150 MB) — same as the segmentation
pass. Use the conda automl env.
"""
import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from decouple import config
from PIL import Image, ImageDraw

from scraper.writer import migrate_db, open_db
from processor.segmentation_worker import (
    download_image,
    get_visual_source_url,
    segment_with_steps,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_DIR = ROOT / "media" / "showcase"
MEDIA_URL_PREFIX = "/media/showcase"
DISPLAY_MAX = 600          # cap longest edge of rendered step images
MASK_COLOR = (0, 184, 148)  # teal overlay
BBOX_COLOR = (231, 76, 60)  # red detection box

STEP_LABELS = [
    "1. Original photo",
    "2. YOLOv8 detection",
    "3. SAM mask",
    "4. White-bg crop (CLIP input)",
]


def _fit(image: Image.Image, max_side: int = DISPLAY_MAX) -> Image.Image:
    """Downscale so the longest edge is <= max_side (never upscales)."""
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return image.copy()
    return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def _render_detect(src: Image.Image, bbox: list[float], disp: Image.Image) -> Image.Image:
    """Source (display size) with the YOLO bbox drawn, coords scaled to disp."""
    sx, sy = disp.width / src.width, disp.height / src.height
    out = disp.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    draw.rectangle(
        [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
        outline=BBOX_COLOR,
        width=max(2, round(disp.width / 150)),
    )
    return out


def _render_mask(mask: np.ndarray, disp: Image.Image) -> Image.Image:
    """Overlay the SAM mask (semi-transparent colour) on the display image."""
    mask_img = Image.fromarray((mask * 255).astype("uint8"), mode="L").resize(
        disp.size, Image.NEAREST
    )
    color = Image.new("RGB", disp.size, MASK_COLOR)
    tinted = Image.blend(disp.convert("RGB"), color, 0.5)
    return Image.composite(tinted, disp.convert("RGB"), mask_img)


def _candidate_ids(conn, count: int, seed: int | None) -> list[str]:
    """Random offers whose visual segmentation succeeded — a few extra so we can
    skip any that fail to re-download/segment at render time. A ``seed`` makes the
    pick reproducible (drawn in Python from the full id list); otherwise SQLite's
    RANDOM() shuffles a fresh selection each run."""
    pool = count * 5
    if seed is not None:
        ids = [
            r[0] for r in conn.execute(
                "SELECT offer_id FROM offer_features WHERE segmentation_reason = 'ok' "
                "ORDER BY offer_id"
            ).fetchall()
        ]
        random.seed(seed)
        return random.sample(ids, min(pool, len(ids)))
    rows = conn.execute(
        "SELECT offer_id FROM offer_features WHERE segmentation_reason = 'ok' "
        "ORDER BY RANDOM() LIMIT ?",
        (pool,),
    ).fetchall()
    return [r[0] for r in rows]


def _title(conn, offer_id: str) -> str:
    row = conn.execute("SELECT title FROM offers WHERE id = ?", (offer_id,)).fetchone()
    return (row[0] if row and row[0] else offer_id)[:80]


def build_showcase(conn, count: int = 3, seed: int | None = None) -> list[dict]:
    # Reset our own artifacts so stale offers don't linger — but leave sibling
    # manifests (e.g. the spec showcase's) untouched.
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    for child in SHOWCASE_DIR.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
    (SHOWCASE_DIR / "manifest.json").unlink(missing_ok=True)

    candidates = _candidate_ids(conn, count, seed)

    manifest: list[dict] = []
    for offer_id in candidates:
        if len(manifest) >= count:
            break
        url = get_visual_source_url(conn, offer_id)
        if not url:
            continue
        src = download_image(url)
        if src is None:
            continue
        steps = segment_with_steps(src)
        if not steps["success"]:
            log.info("skip %s (segmentation reason=%s)", offer_id, steps["reason"])
            continue

        disp = _fit(src)
        renders = [
            disp,
            _render_detect(src, steps["bbox"], disp),
            _render_mask(steps["mask"], disp),
            _fit(steps["crop"]),
        ]

        out_dir = SHOWCASE_DIR / offer_id
        out_dir.mkdir(parents=True, exist_ok=True)
        entry_steps = []
        for i, (label, img) in enumerate(zip(STEP_LABELS, renders), start=1):
            fname = f"{i}_{['original', 'detect', 'mask', 'crop'][i - 1]}.jpg"
            img.convert("RGB").save(out_dir / fname, "JPEG", quality=85)
            entry_steps.append({"label": label, "url": f"{MEDIA_URL_PREFIX}/{offer_id}/{fname}"})

        manifest.append({"offer_id": offer_id, "title": _title(conn, offer_id), "steps": entry_steps})
        log.info("rendered showcase for %s", offer_id)

    (SHOWCASE_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cluster-lab image-processing showcase")
    parser.add_argument("--count", type=int, default=3, help="number of bikes to showcase")
    parser.add_argument("--seed", type=int, default=None, help="reproducible selection")
    args = parser.parse_args()

    conn = open_db(config("DB_PATH"))
    migrate_db(conn)
    manifest = build_showcase(conn, count=args.count, seed=args.seed)
    conn.close()
    log.info("Showcase built: %d bikes → %s", len(manifest), SHOWCASE_DIR / "manifest.json")


if __name__ == "__main__":
    main()
