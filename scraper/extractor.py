import json
import re
from urllib.parse import parse_qs, unquote, urlparse


def extract_store_state(html: str) -> dict:
    """
    Find and parse the __listing_StoreState JSON blob from Allegro page HTML.

    Two formats are handled:
    - Current: {"__listing_StoreState": {...}} in a data attribute
    - Legacy:  window.__listing_StoreState = {...} in an inline script
    """
    # Current format: embedded as a JSON object value in a data attribute
    idx = html.find('{"__listing_StoreState":')
    if idx != -1:
        try:
            outer, _ = json.JSONDecoder().raw_decode(html, idx)
            state = outer.get("__listing_StoreState")
            if state and isinstance(state, dict):
                return state
        except json.JSONDecodeError:
            pass

    # Legacy format: inline JS assignment
    match = re.search(
        r'window\.__listing_StoreState\s*=\s*(\{.*?)\s*(?:;?\s*</script>|;?\s*window\.)',
        html,
        re.DOTALL,
    )
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse __listing_StoreState JSON: {exc}") from exc

    raise ValueError("__listing_StoreState not found in page HTML")


def get_last_page(state: dict) -> int:
    """Return lastAvailablePage from searchMeta, falling back to 1."""
    try:
        # Current format: searchMeta lives inside items
        return int(state["items"]["searchMeta"]["lastAvailablePage"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        # Legacy format: searchMeta at top level
        return int(state["searchMeta"]["lastAvailablePage"])
    except (KeyError, TypeError, ValueError):
        return 1


def parse_offers(state: dict) -> list[dict]:
    """
    Normalise raw offer dicts from the state blob into the format expected
    by downloader.writer.upsert_offer / upsert_parameters / upsert_images.
    """
    items_block = state.get("items") or {}

    # Current format: flat elements list
    raw = list(items_block.get("elements") or [])

    # Legacy format: promoted + regular split
    if not raw:
        raw = list(items_block.get("promoted") or []) + list(items_block.get("regular") or [])

    result: list[dict] = []
    for item in raw:
        # Current format uses "offerId"; legacy used "id"
        offer_id = str(item.get("offerId") or item.get("id") or "")
        if not offer_id:
            continue

        # Title: current = {text: ...}, legacy = plain string
        title_obj = item.get("title")
        if isinstance(title_obj, dict):
            title = title_obj.get("text") or ""
        else:
            title = str(title_obj or item.get("name") or "")

        # URL: may be absolute, root-relative, or a tracking redirect
        url: str = item.get("url") or ""
        if url and not url.startswith("http"):
            url = f"https://allegro.pl{url}"
        # Unwrap /events/clicks tracking redirects → real offer URL
        if "/events/clicks" in url:
            qs = parse_qs(urlparse(url).query)
            redirect = qs.get("redirect", [""])[0]
            if redirect:
                url = unquote(redirect)

        # Price: current = price.mainPrice.{amount, currency}
        #        legacy  = sellingMode.price.{amount, currency}
        price_obj: dict = {}
        price_block = item.get("price") or {}
        if "mainPrice" in price_block:
            price_obj = price_block["mainPrice"]
        elif "amount" in price_block:
            price_obj = price_block
        else:
            price_obj = (item.get("sellingMode") or {}).get("price") or {}

        # Images: current = photos[{small, medium}], legacy = images[{url}]
        photos = item.get("photos") or item.get("images") or []
        images: list[dict] = []
        for photo in photos:
            url_val = (
                photo.get("medium")
                or photo.get("small")
                or photo.get("url")
                or ""
            )
            if url_val:
                images.append({"url": url_val})

        # Parameters: current = [{name, values: [str, ...]}, ...]
        #             legacy  = [{name, value: str}, ...]
        raw_params = item.get("parameters") or item.get("generalParameters") or []
        params: list[dict] = []
        for p in raw_params:
            name = p.get("name") or ""
            if not name:
                continue
            values = p.get("values")
            if isinstance(values, list):
                value = ", ".join(str(v) for v in values)
            else:
                value = str(p.get("value") or "")
            params.append({"name": name, "value": value})

        # Condition may be absent in listing view
        condition = item.get("condition")
        listing_type = item.get("context") or (item.get("sellingMode") or {}).get("format")

        result.append({
            "id": offer_id,
            "name": title,
            "sellingMode": {
                "format": listing_type,
                "price": price_obj,
            },
            "seller": item.get("seller") or {},
            "condition": condition,
            "url": url,
            "thumbnail_url": item.get("mainThumbnail") or "",
            "images": images,
            "parameters": params,
            "publication": item.get("publication") or {},
        })
    return result
