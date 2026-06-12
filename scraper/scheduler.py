import argparse
import logging
import os
import random
import sys
import time
from urllib.parse import quote_plus

from decouple import config

from downloader.writer import init_db, open_db, upsert_images, upsert_offer, upsert_parameters
from scraper.browser import new_driver
from scraper.extractor import extract_store_state, get_last_page, parse_offers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_BASE = "https://allegro.pl/kategoria/rowery-i-akcesoria-rowery-16420"
_HOME = "https://allegro.pl"


def _page_url(search: str, page_num: int) -> str:
    return f"{_BASE}?string={quote_plus(search)}&p={page_num}"


def _human_scroll(driver) -> None:
    total = random.randint(400, 900)
    step = random.randint(80, 160)
    for offset in range(step, total, step):
        driver.execute_script(f"window.scrollTo({{top: {offset}, behavior: 'smooth'}})")
        time.sleep(random.uniform(0.06, 0.18))


def _accept_cookies(driver) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    try:
        btn = WebDriverWait(driver, 8).until(
            EC.element_to_be_clickable((By.XPATH,
                "//button[contains(text(),'Akceptuję') or "
                "contains(text(),'Akceptuj wszystkie') or "
                "contains(text(),'Zezwól na wszystkie') or "
                "contains(text(),'Zgadzam się')]"
            ))
        )
        btn.click()
        time.sleep(random.uniform(0.5, 1.5))
        log.info("Cookie consent accepted")
    except Exception:
        log.debug("No cookie consent popup detected — continuing")


def _warm_up(driver) -> None:
    log.info("Warm-up: visiting Allegro homepage")
    driver.get(_HOME)
    time.sleep(random.uniform(2.5, 5.0))
    _accept_cookies(driver)
    _human_scroll(driver)
    time.sleep(random.uniform(1.0, 2.5))


def scrape_page(driver, page_num: int, search: str) -> tuple[dict, list[dict]]:
    url = _page_url(search, page_num)
    driver.get(url)

    # Give the page time to execute JS and populate __listing_StoreState
    deadline = time.time() + 20
    while time.time() < deadline:
        has_state = driver.execute_script(
            "return typeof window.__listing_StoreState !== 'undefined'"
        )
        if has_state:
            break
        time.sleep(0.5)

    _human_scroll(driver)

    # Primary: read directly from JS runtime (avoids regex on minified HTML)
    try:
        state = driver.execute_script("return window.__listing_StoreState")
        if state and isinstance(state, dict):
            return state, parse_offers(state)
    except Exception:
        pass

    # Fallback: parse the HTML source
    html = driver.page_source
    state = extract_store_state(html)
    return state, parse_offers(state)


def _save(conn, offer: dict) -> None:
    offer_id = offer.get("id")
    if not offer_id:
        return
    upsert_offer(conn, offer)
    upsert_parameters(conn, offer_id, offer.get("parameters", []))
    upsert_images(conn, offer_id, offer.get("images", []))


def run(max_pages: int | None = None) -> int:
    db_path = config("DB_PATH")
    search = config("SCRAPER_SEARCH", default="szosowy")
    min_delay = config("SCRAPER_MIN_DELAY", default=3.0, cast=float)
    max_delay = config("SCRAPER_MAX_DELAY", default=7.0, cast=float)

    conn = open_db(db_path)
    init_db(conn)
    total = 0

    driver = new_driver()
    try:
        _warm_up(driver)

        log.info("Scraping page 1 (%s)", _page_url(search, 1))
        state, offers = scrape_page(driver, 1, search)
        last_page = get_last_page(state)
        if max_pages is not None:
            last_page = min(last_page, max_pages)
        log.info("Total pages to scrape: %d", last_page)

        for offer in offers:
            _save(conn, offer)
        conn.commit()
        total += len(offers)
        log.info("Page 1/%d saved: %d offers", last_page, len(offers))

        for page_num in range(2, last_page + 1):
            delay = random.uniform(min_delay, max_delay)
            log.info("Waiting %.1fs before page %d", delay, page_num)
            time.sleep(delay)

            log.info("Scraping page %d/%d", page_num, last_page)
            try:
                _, offers = scrape_page(driver, page_num, search)
            except Exception as exc:
                log.warning("Page %d failed: %s — skipping", page_num, exc)
                continue

            for offer in offers:
                _save(conn, offer)
            conn.commit()
            total += len(offers)
            log.info(
                "Page %d/%d saved: %d offers (running total: %d)",
                page_num, last_page, len(offers), total,
            )
    finally:
        driver.quit()

    conn.close()
    log.info("Done. Total offers written: %d", total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Allegro scraper (undetected Firefox)")
    parser.add_argument("--max-pages", type=int, default=None, help="Cap page count")
    args = parser.parse_args()

    max_p = args.max_pages
    if not max_p:
        env_max = config("SCRAPER_MAX_PAGES", default="").strip()
        if env_max.isdigit():
            max_p = int(env_max)

    run(max_pages=max_p)


if __name__ == "__main__":
    _BASE = os.getenv("BASE_URL")
    _HOME = os.getenv("HOME_URL")
    main()
