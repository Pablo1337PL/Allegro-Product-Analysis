import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scraper.extractor import extract_store_state, get_last_page, parse_offers

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_STATE: dict = {
    "items": {
        "promoted": [
            {
                "id": "promo-001",
                "name": "Promoted Bike",
                "price": {"amount": "4999.00", "currency": "PLN"},
                "seller": {"id": "s1", "login": "seller_pro"},
                "condition": "NEW",
                "url": "/oferta/promoted-bike-promo-001",
                "images": [{"url": "https://img.example.com/promo.jpg"}],
                "parameters": [{"name": "Rozmiar", "values": ["L"]}],
            }
        ],
        "regular": [
            {
                "id": "reg-001",
                "name": "Carbon Road Bike",
                "price": {"amount": "2500.00", "currency": "PLN"},
                "seller": {"id": "s2", "login": "seller_reg"},
                "condition": "USED",
                "url": "https://allegro.pl/oferta/carbon-bike-reg-001",
                "images": [
                    {"url": "https://img.example.com/a.jpg"},
                    {"url": "https://img.example.com/b.jpg"},
                ],
                "parameters": [
                    {"name": "Rozmiar ramy", "values": ["M"]},
                    {"name": "Materiał ramy", "values": ["Karbon"]},
                ],
            },
            {
                "id": "reg-002",
                "name": "Aluminium Road Bike",
                "price": {"amount": "1200.00", "currency": "PLN"},
                "seller": {"id": "s3", "login": "seller_two"},
                "condition": "USED",
                "url": "/oferta/alu-bike-reg-002",
                "images": [],
                "parameters": [],
            },
        ],
    },
    "searchMeta": {
        "lastAvailablePage": 42,
        "totalCount": 2520,
    },
}


def _make_html(state: dict) -> str:
    blob = json.dumps(state)
    return f"""
    <html><head></head><body>
    <script>
    var x = 1;
    window.__listing_StoreState={blob};
    window.__other = {{}};
    </script>
    </body></html>
    """


# ── extractor: extract_store_state ────────────────────────────────────────────


def test_extract_store_state_finds_blob():
    html = _make_html(SAMPLE_STATE)
    result = extract_store_state(html)
    assert result["searchMeta"]["lastAvailablePage"] == 42
    assert len(result["items"]["regular"]) == 2


def test_extract_store_state_raises_on_missing():
    with pytest.raises(ValueError, match="not found"):
        extract_store_state("<html><body>no state here</body></html>")


def test_extract_store_state_raises_on_malformed_json():
    bad_html = "<script>window.__listing_StoreState = {broken: json;</script>"
    with pytest.raises(ValueError):
        extract_store_state(bad_html)


# ── extractor: get_last_page ──────────────────────────────────────────────────


def test_get_last_page_returns_correct_value():
    assert get_last_page(SAMPLE_STATE) == 42


def test_get_last_page_defaults_to_one_when_missing():
    assert get_last_page({}) == 1
    assert get_last_page({"searchMeta": {}}) == 1


def test_get_last_page_handles_non_int():
    assert get_last_page({"searchMeta": {"lastAvailablePage": "not_a_number"}}) == 1


# ── extractor: parse_offers ───────────────────────────────────────────────────


def test_parse_offers_returns_all_items():
    offers = parse_offers(SAMPLE_STATE)
    assert len(offers) == 3  # 1 promoted + 2 regular


def test_parse_offers_normalises_price_to_selling_mode():
    offers = parse_offers(SAMPLE_STATE)
    reg = next(o for o in offers if o["id"] == "reg-001")
    assert reg["sellingMode"]["price"]["amount"] == "2500.00"
    assert reg["sellingMode"]["price"]["currency"] == "PLN"


def test_parse_offers_expands_relative_urls():
    offers = parse_offers(SAMPLE_STATE)
    promo = next(o for o in offers if o["id"] == "promo-001")
    assert promo["url"].startswith("https://allegro.pl")


def test_parse_offers_preserves_absolute_urls():
    offers = parse_offers(SAMPLE_STATE)
    reg = next(o for o in offers if o["id"] == "reg-001")
    assert reg["url"] == "https://allegro.pl/oferta/carbon-bike-reg-001"


def test_parse_offers_returns_empty_on_empty_state():
    assert parse_offers({}) == []
    assert parse_offers({"items": {}}) == []
    assert parse_offers({"items": {"promoted": [], "regular": []}}) == []


def test_parse_offers_skips_items_without_id():
    state = {"items": {"regular": [{"name": "No ID Bike", "price": {}}]}}
    assert parse_offers(state) == []


def test_parse_offers_images_and_parameters():
    offers = parse_offers(SAMPLE_STATE)
    reg = next(o for o in offers if o["id"] == "reg-001")
    assert len(reg["images"]) == 2
    assert len(reg["parameters"]) == 2
    assert reg["parameters"][0]["name"] == "Rozmiar ramy"


# ── scheduler: scrape_page ────────────────────────────────────────────────────


def test_scrape_page_uses_evaluate_primary():
    from scraper.scheduler import scrape_page

    mock_page = AsyncMock()
    mock_page.evaluate.return_value = SAMPLE_STATE

    state, offers = asyncio.run(scrape_page(mock_page, 1, "rower szosowy"))

    mock_page.goto.assert_called_once()
    assert "searchMeta" in state
    assert len(offers) == 3


def test_scrape_page_falls_back_to_html():
    from scraper.scheduler import scrape_page

    mock_page = AsyncMock()
    mock_page.evaluate.side_effect = Exception("JS error")
    mock_page.content.return_value = _make_html(SAMPLE_STATE)

    state, offers = asyncio.run(scrape_page(mock_page, 1, "rower szosowy"))

    assert len(offers) == 3


def test_scrape_page_url_includes_page_number():
    from scraper.scheduler import scrape_page

    mock_page = AsyncMock()
    mock_page.evaluate.return_value = SAMPLE_STATE

    asyncio.run(scrape_page(mock_page, 3, "rower szosowy"))

    call_args = mock_page.goto.call_args
    called_url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
    assert "p=3" in called_url


# ── scheduler: run ────────────────────────────────────────────────────────────


class _AsyncCM:
    """Minimal async context manager that returns a given value."""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_):
        pass


def _make_mock_page(state: dict) -> AsyncMock:
    page = AsyncMock()
    page.evaluate.return_value = state
    page.context = AsyncMock()
    page.context.browser = AsyncMock()
    page.context.browser.close = AsyncMock()
    return page


def test_run_writes_offers_to_db(tmp_path, monkeypatch):
    from scraper import scheduler

    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setenv("SCRAPER_MIN_DELAY", "0")
    monkeypatch.setenv("SCRAPER_MAX_DELAY", "0")

    single_page_state = {
        **SAMPLE_STATE,
        "searchMeta": {"lastAvailablePage": 1},
    }
    mock_page = _make_mock_page(single_page_state)
    mock_pw = MagicMock()

    with patch("scraper.scheduler.async_playwright", return_value=_AsyncCM(mock_pw)), \
         patch("scraper.scheduler.new_page", new=AsyncMock(return_value=mock_page)), \
         patch("scraper.scheduler.asyncio.sleep", new=AsyncMock()):
        count = asyncio.run(scheduler.run(max_pages=1))

    assert count == 3
    conn = sqlite3.connect(db_file)
    rows = conn.execute("SELECT id FROM offers").fetchall()
    conn.close()
    assert len(rows) == 3


def test_run_respects_max_pages(tmp_path, monkeypatch):
    from scraper import scheduler

    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setenv("SCRAPER_MIN_DELAY", "0")
    monkeypatch.setenv("SCRAPER_MAX_DELAY", "0")

    # State says 50 pages but we cap at 2
    big_state = {**SAMPLE_STATE, "searchMeta": {"lastAvailablePage": 50}}
    mock_page = _make_mock_page(big_state)
    mock_pw = MagicMock()

    with patch("scraper.scheduler.async_playwright", return_value=_AsyncCM(mock_pw)), \
         patch("scraper.scheduler.new_page", new=AsyncMock(return_value=mock_page)), \
         patch("scraper.scheduler.asyncio.sleep", new=AsyncMock()):
        count = asyncio.run(scheduler.run(max_pages=2))

    # 3 offers per page × 2 pages = 6, but upsert is idempotent so 3 unique rows
    assert count == 6
    conn = sqlite3.connect(db_file)
    total = conn.execute("SELECT COUNT(*) FROM offers").fetchone()[0]
    conn.close()
    assert total == 3  # same 3 offers upserted twice


def test_run_continues_on_page_error(tmp_path, monkeypatch):
    from scraper import scheduler

    db_file = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_file)
    monkeypatch.setenv("SCRAPER_MIN_DELAY", "0")
    monkeypatch.setenv("SCRAPER_MAX_DELAY", "0")

    two_page_state = {**SAMPLE_STATE, "searchMeta": {"lastAvailablePage": 2}}
    call_count = {"n": 0}

    async def fake_scrape_page(page, page_num, search):
        call_count["n"] += 1
        if page_num == 1:
            return two_page_state, parse_offers(two_page_state)
        raise RuntimeError("Network error on page 2")

    mock_page = _make_mock_page(two_page_state)
    mock_pw = MagicMock()

    with patch("scraper.scheduler.async_playwright", return_value=_AsyncCM(mock_pw)), \
         patch("scraper.scheduler.new_page", new=AsyncMock(return_value=mock_page)), \
         patch("scraper.scheduler.scrape_page", new=fake_scrape_page), \
         patch("scraper.scheduler.asyncio.sleep", new=AsyncMock()):
        count = asyncio.run(scheduler.run(max_pages=2))

    # Page 2 failed but page 1 should have written 3 offers
    assert count == 3
