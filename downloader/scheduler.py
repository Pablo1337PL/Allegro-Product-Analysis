import argparse
import logging
import os
import sys

from downloader.client import get_offer_detail, iter_all_offers
from downloader.writer import init_db, open_db, upsert_images, upsert_offer, upsert_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def run_full_download(max_pages: int | None = None, dry_run: bool = False) -> int:
    db_path = os.environ["DB_PATH"]
    log.info("Opening DB at %s", db_path)
    conn = open_db(db_path)
    init_db(conn)

    total = 0
    for summary in iter_all_offers(max_pages=max_pages):
        offer_id = summary.get("id")
        if not offer_id:
            continue

        if dry_run:
            log.info("[DRY RUN] %s — %s", offer_id, summary.get("name") or summary.get("title"))
            total += 1
            continue

        try:
            detail = get_offer_detail(offer_id)
        except Exception as exc:
            log.warning("Detail fetch failed for %s: %s — using summary data", offer_id, exc)
            detail = summary

        upsert_offer(conn, detail)
        upsert_parameters(conn, offer_id, detail.get("parameters", []))
        upsert_images(conn, offer_id, detail.get("images", []))
        conn.commit()

        total += 1
        if total % 50 == 0:
            log.info("Progress: %d offers saved", total)

    log.info("Done. Total offers processed: %d", total)
    conn.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Allegro road bikes downloader")
    parser.add_argument("--full", action="store_true", help="Run full download")
    parser.add_argument("--dry-run", action="store_true", help="List without saving")
    parser.add_argument("--max-pages", type=int, default=None, help="Cap number of pages")
    args = parser.parse_args()

    if args.full or args.dry_run:
        run_full_download(max_pages=args.max_pages, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
