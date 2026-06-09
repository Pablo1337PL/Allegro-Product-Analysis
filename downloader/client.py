import os
import time
from typing import Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from downloader.auth import get_token

ROAD_BIKES_CATEGORY_ID = "16484"
_RETRY_ON = (httpx.HTTPStatusError, httpx.RequestError)


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {get_token()}",
        "Accept": "application/vnd.allegro.public.v1+json",
    }


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(_RETRY_ON),
    reraise=True,
)
def _search_page(offset: int, limit: int = 100) -> list[dict]:
    base = os.environ.get("ALLEGRO_API_BASE", "https://api.allegro.pl")
    sleep_s = float(os.environ.get("DOWNLOAD_SLEEP_SECONDS", "0.5"))
    phrase = os.environ.get("SEARCH_QUERY", "rower szosowy")
    params: dict = {"category.id": ROAD_BIKES_CATEGORY_ID, "limit": limit, "offset": offset}
    if phrase:
        params["phrase"] = phrase
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{base}/offers/listing", headers=_headers(), params=params)
        if not resp.is_success:
            import logging
            logging.getLogger(__name__).error(
                "API error %s for %s — body: %s", resp.status_code, resp.url, resp.text[:500]
            )
        resp.raise_for_status()
    time.sleep(sleep_s)
    data = resp.json()
    items = data.get("items", {})
    return items.get("promoted", []) + items.get("regular", [])


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type(_RETRY_ON),
    reraise=True,
)
def get_offer_detail(offer_id: str) -> dict:
    base = os.environ.get("ALLEGRO_API_BASE", "https://api.allegro.pl")
    sleep_s = float(os.environ.get("DOWNLOAD_SLEEP_SECONDS", "0.5"))
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{base}/offers/{offer_id}", headers=_headers())
        resp.raise_for_status()
    time.sleep(sleep_s)
    return resp.json()


def iter_all_offers(max_pages: int | None = None) -> Iterator[dict]:
    limit = 100
    offset = 0
    page = 0
    while True:
        if max_pages is not None and page >= max_pages:
            break
        offers = _search_page(offset=offset, limit=limit)
        if not offers:
            break
        yield from offers
        if len(offers) < limit:
            break
        offset += limit
        page += 1
