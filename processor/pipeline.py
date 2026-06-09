import logging
import os
import pickle
import sys

from downloader.writer import open_db
from processor.clip_worker import CLIPWorker
from processor.scraper import DescriptionScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BATCH_SIZE = 10


def run_pipeline() -> int:
    db_path = os.environ["DB_PATH"]
    conn = open_db(db_path)

    rows = conn.execute(
        """
        SELECT o.id, o.offer_url
        FROM offers o
        LEFT JOIN offer_features f ON o.id = f.offer_id
        WHERE f.offer_id IS NULL
        ORDER BY o.fetched_at DESC
        """
    ).fetchall()

    log.info("Found %d unprocessed offers", len(rows))
    if not rows:
        conn.close()
        return 0

    clip = CLIPWorker()
    scraper = DescriptionScraper()
    processed = 0

    try:
        for offer_id, offer_url in rows:
            img_row = conn.execute(
                "SELECT url FROM offer_images WHERE offer_id=? AND position=0",
                (offer_id,),
            ).fetchone()
            image_url = img_row[0] if img_row else None

            description = ""
            if offer_url:
                description = scraper.scrape_description(offer_url)

            clip_bytes = None
            if image_url:
                vector = clip.encode_image_url(image_url)
                if vector is not None:
                    clip_bytes = pickle.dumps(vector)

            conn.execute(
                """
                INSERT INTO offer_features (offer_id, description_text, clip_vector)
                VALUES (?, ?, ?)
                ON CONFLICT(offer_id) DO UPDATE SET
                    description_text = excluded.description_text,
                    clip_vector = excluded.clip_vector
                """,
                (offer_id, description, clip_bytes),
            )
            processed += 1

            if processed % BATCH_SIZE == 0:
                conn.commit()
                log.info("Processed %d / %d", processed, len(rows))

    finally:
        scraper.close()

    conn.commit()
    conn.close()
    log.info("Pipeline complete. Processed %d offers.", processed)
    return processed


if __name__ == "__main__":
    run_pipeline()
