"""Multi-pass feature processor.

Pass 1 — CLIP embedding:   offers without clip_vector → download image → embed → store
Pass 2 — Zero-shot attrs:  offers with clip_vector but no zero-shot data → classify
Pass 3 — Condition:        grade condition 1-5 + enrich spec_stan from description (Ollama)
Pass 4 — Segmentation:     is_bike offers → YOLO+SAM crop → clip_vector_visual (Phase B)

Groupset/tier/frame-year extraction and spec_* backfill from description text now
happen inline during the detail scrape (scraper/detail_scheduler.py's
OllamaExtractorWorker), not here — see CLAUDE.md.

Run all passes:   python processor/pipeline.py
Run one pass:     python processor/pipeline.py --passes clip
                  python processor/pipeline.py --passes zero-shot
                  python processor/pipeline.py --passes condition
                  python processor/pipeline.py --passes segmentation
"""
import argparse
import logging
import os
import pickle
import sys
from datetime import datetime, timezone

from decouple import config

from scraper.writer import apply_zero_shot, migrate_db, open_db, upsert_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_BATCH = 50


def _pass_clip(conn) -> int:
    """Pass 1: encode images for offers that have no clip_vector yet.

    Embeds the first full-resolution gallery image (is_thumbnail=0, lowest
    position) when one exists, falling back to the small listing thumbnail only
    for offers that have not been detail-scraped yet.
    """
    rows = conn.execute(
        """
        SELECT o.id, o.description
        FROM offers o
        LEFT JOIN offer_features f ON o.id = f.offer_id
        WHERE f.clip_vector IS NULL
          AND o.is_bike = 1
        ORDER BY o.fetched_at DESC
        """
    ).fetchall()

    log.info("[clip] %d offers need embedding", len(rows))
    if not rows:
        return 0

    from processor.clip_worker import CLIPWorker
    clip = CLIPWorker()
    processed = 0

    for offer_id, description in rows:
        img_row = conn.execute(
            """
            SELECT url FROM offer_images
            WHERE offer_id = ?
            ORDER BY is_thumbnail ASC, position ASC
            LIMIT 1
            """,
            (offer_id,),
        ).fetchone()
        image_url = img_row[0] if img_row else None

        clip_bytes = None
        if image_url:
            vector = clip.encode_image_url(image_url)
            if vector is not None:
                clip_bytes = pickle.dumps(vector)

        upsert_features(conn, offer_id, {
            "clip_vector": clip_bytes,
            "description_text": description or "",
        })
        processed += 1

        if processed % _BATCH == 0:
            conn.commit()
            log.info("[clip] %d / %d", processed, len(rows))

    conn.commit()
    log.info("[clip] done — %d offers processed", processed)
    return processed


def _pass_zero_shot(conn) -> int:
    """Pass 2: run zero-shot attribute detection for offers that have a
    clip_vector but no zero-shot data yet."""
    rows = conn.execute(
        """
        SELECT f.offer_id, f.clip_vector
        FROM offer_features f
        JOIN offers o ON o.id = f.offer_id
        WHERE f.clip_vector IS NOT NULL
          AND o.has_drop_handlebars IS NULL
          AND o.is_bike = 1
        """
    ).fetchall()

    log.info("[zero-shot] %d offers need attribute detection", len(rows))
    if not rows:
        return 0

    from processor.clip_worker import CLIPWorker
    clip = CLIPWorker()
    processed = 0

    for offer_id, clip_bytes in rows:
        try:
            vector = pickle.loads(clip_bytes)
        except Exception as exc:
            log.warning("[zero-shot] bad vector for %s: %s", offer_id, exc)
            continue

        attrs = clip.run_zero_shot(vector)
        apply_zero_shot(conn, offer_id, attrs)
        processed += 1

        if processed % _BATCH == 0:
            conn.commit()
            log.info("[zero-shot] %d / %d", processed, len(rows))

    conn.commit()
    log.info("[zero-shot] done — %d offers processed", processed)
    return processed


def _pass_segmentation(conn) -> int:
    """Pass 4: segment the bike out of the detail image and embed the crop into
    clip_vector_visual. Runs for is_bike=1 offers not yet segmented."""
    rows = conn.execute(
        """
        SELECT o.id
        FROM offers o
        LEFT JOIN offer_features f ON o.id = f.offer_id
        WHERE o.is_bike = 1
          AND (f.segmented_at IS NULL)
        ORDER BY o.fetched_at DESC
        """
    ).fetchall()

    log.info("[segmentation] %d offers need a visual embedding", len(rows))
    if not rows:
        return 0

    conf = config("SEGMENTATION_CONF_THRESHOLD", default=0.5, cast=float)
    batch = config("SEGMENTATION_BATCH_SIZE", default=25, cast=int)

    from processor.clip_worker import CLIPWorker
    from processor import segmentation_worker as seg
    clip = CLIPWorker()

    media_dir = "media/segmented"
    import os
    os.makedirs(media_dir, exist_ok=True)
    now = lambda: datetime.now(timezone.utc).isoformat()
    processed = 0

    for (offer_id,) in rows:
        url = seg.get_visual_source_url(conn, offer_id)
        image = seg.download_image(url) if url else None
        if image is None:
            upsert_features(conn, offer_id, {
                "segmentation_success": 0,
                "segmentation_reason": "download_failed",
                "segmented_at": now(),
            })
            processed += 1
            continue

        result = seg.segment_bike(image, conf_threshold=conf)
        path = None
        if result["success"]:
            embed_source = result["segmented_image"]
            path = f"{media_dir}/{offer_id}.jpg"
            embed_source.save(path, "JPEG", quality=90)
        elif result["reason"] in ("no_mask", "mask_too_small") and result["bbox"]:
            embed_source = seg.crop_to_bbox(image, result["bbox"])
        else:
            embed_source = image  # no_detection → embed the whole image

        vector = clip.encode_image(embed_source)
        upsert_features(conn, offer_id, {
            "clip_vector_visual": pickle.dumps(vector),
            "segmentation_success": 1 if result["success"] else 0,
            "segmentation_reason": result["reason"],
            "segmented_image_path": path,
            "segmented_at": now(),
        })
        processed += 1

        if processed % batch == 0:
            conn.commit()
            log.info("[segmentation] %d / %d", processed, len(rows))

    conn.commit()
    log.info("[segmentation] done — %d offers processed", processed)
    return processed


def _pass_condition(conn) -> int:
    """Grade condition (1-5) from the description into offers.condition, and
    enrich spec_stan (fill a NULL, or mark 'Uszkodzony' when the description shows
    damage). Needs Ollama — skipped if unreachable. Idempotent via
    condition_extracted_at."""
    from processor.condition_worker import classify_condition
    from scraper.ollama_common import ollama_available
    from scraper.writer import mark_condition_done, update_condition

    rows = conn.execute(
        """
        SELECT id, title, description, spec_stan FROM offers
        WHERE is_bike = 1 AND description IS NOT NULL AND length(description) > 0
          AND condition_extracted_at IS NULL
        """
    ).fetchall()
    log.info("[condition] %d offers to grade", len(rows))
    if not rows:
        return 0
    if not ollama_available():
        log.warning("[condition] Ollama unavailable — skipping (needs the LLM)")
        return 0

    debug_csv = os.environ.get("CONDITION_DEBUG_CSV") or None
    if debug_csv:
        log.info("[condition] debug CSV → %s", debug_csv)
    batch = config("BACKFILL_BATCH_SIZE", default=25, cast=int)
    processed = graded = 0

    for offer_id, title, description, spec_stan in rows:
        res = classify_condition(title, description, spec_stan, debug_csv=debug_csv)
        if res is None:
            mark_condition_done(conn, offer_id)
        else:
            # Enrich spec_stan: damage overrides any declared state; otherwise only
            # fill a missing one — never relabel a seller's Nowy/Używany.
            new_stan = None
            if res["broken"] and (spec_stan or "").strip().lower() != "uszkodzony":
                new_stan = "Uszkodzony"
            elif spec_stan is None and res["stan"]:
                new_stan = res["stan"]
            update_condition(conn, offer_id, str(res["grade"]), spec_stan=new_stan)
            graded += 1
        processed += 1
        if processed % batch == 0:
            conn.commit()
            log.info("[condition] %d / %d (%d graded)", processed, len(rows), graded)

    conn.commit()
    log.info("[condition] done — %d processed, %d graded", processed, graded)
    return graded


def run_pipeline(passes: list[str] | None = None) -> dict[str, int]:
    db_path = config("DB_PATH")
    conn = open_db(db_path)
    migrate_db(conn)

    all_passes = ["clip", "zero-shot", "condition", "segmentation"]
    selected = passes if passes else all_passes

    results: dict[str, int] = {}
    for p in selected:
        if p == "clip":
            results["clip"] = _pass_clip(conn)
        elif p == "zero-shot":
            results["zero-shot"] = _pass_zero_shot(conn)
        elif p == "condition":
            results["condition"] = _pass_condition(conn)
        elif p == "segmentation":
            results["segmentation"] = _pass_segmentation(conn)
        else:
            log.warning("Unknown pass %r — skipping", p)

    conn.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature processor pipeline")
    parser.add_argument(
        "--passes",
        default=None,
        help="Comma-separated passes: clip,zero-shot,condition,segmentation "
             "(default: all)",
    )
    args = parser.parse_args()
    passes = [p.strip() for p in args.passes.split(",")] if args.passes else None
    results = run_pipeline(passes=passes)
    for name, count in results.items():
        log.info("  %-15s → %d offers", name, count)


if __name__ == "__main__":
    main()
