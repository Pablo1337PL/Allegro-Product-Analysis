import json
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

from downloader.writer import init_db, upsert_images, upsert_offer, upsert_parameters

# ── helpers ───────────────────────────────────────────────────────────────────


def mem_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    return conn


SAMPLE = {
    "id": "offer-001",
    "name": "Test Road Bike",
    "condition": "USED",
    "sellingMode": {"format": "BUY_NOW", "price": {"amount": "1299.99", "currency": "PLN"}},
    "seller": {"id": "s-001", "login": "seller_one"},
    "url": "https://allegro.pl/oferta/test-001",
    "images": [
        {"url": "https://img.example.com/a.jpg"},
        {"url": "https://img.example.com/b.jpg"},
    ],
    "publication": {"endingAt": None},
    "parameters": [
        {"name": "Stan", "values": ["używany"]},
        {"name": "Rozmiar ramy", "values": ["M"]},
    ],
}


# ── writer: schema ────────────────────────────────────────────────────────────


def test_init_db_creates_all_tables():
    conn = mem_db()
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"offers", "offer_parameters", "offer_images", "offer_features"} <= tables


def test_init_db_is_idempotent():
    conn = mem_db()
    init_db(conn)  # second call must not raise


# ── writer: offers ────────────────────────────────────────────────────────────


def test_upsert_offer_inserts_row():
    conn = mem_db()
    upsert_offer(conn, SAMPLE)
    conn.commit()
    row = conn.execute("SELECT id, title, price, currency FROM offers WHERE id=?", ("offer-001",)).fetchone()
    assert row is not None
    assert row[0] == "offer-001"
    assert row[2] == pytest.approx(1299.99)
    assert row[3] == "PLN"


def test_upsert_offer_is_idempotent():
    conn = mem_db()
    upsert_offer(conn, SAMPLE)
    upsert_offer(conn, SAMPLE)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM offers").fetchone()[0] == 1


def test_upsert_offer_updates_price():
    conn = mem_db()
    upsert_offer(conn, SAMPLE)
    conn.commit()
    updated = {**SAMPLE, "sellingMode": {"format": "BUY_NOW", "price": {"amount": "999.00", "currency": "PLN"}}}
    upsert_offer(conn, updated)
    conn.commit()
    assert conn.execute("SELECT price FROM offers WHERE id='offer-001'").fetchone()[0] == pytest.approx(999.0)


def test_upsert_offer_stores_raw_json():
    conn = mem_db()
    upsert_offer(conn, SAMPLE)
    conn.commit()
    raw = conn.execute("SELECT raw_json FROM offers WHERE id='offer-001'").fetchone()[0]
    assert json.loads(raw)["id"] == "offer-001"


def test_upsert_offer_skips_missing_id():
    conn = mem_db()
    upsert_offer(conn, {"name": "No ID bike"})
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM offers").fetchone()[0] == 0


# ── writer: parameters ────────────────────────────────────────────────────────


def test_upsert_parameters_inserts_rows():
    conn = mem_db()
    upsert_offer(conn, SAMPLE)
    upsert_parameters(conn, "offer-001", SAMPLE["parameters"])
    conn.commit()
    rows = conn.execute(
        "SELECT name, value FROM offer_parameters WHERE offer_id='offer-001' ORDER BY name"
    ).fetchall()
    names = {r[0] for r in rows}
    assert "Stan" in names and "Rozmiar ramy" in names


def test_upsert_parameters_is_idempotent():
    conn = mem_db()
    upsert_offer(conn, SAMPLE)
    upsert_parameters(conn, "offer-001", SAMPLE["parameters"])
    upsert_parameters(conn, "offer-001", SAMPLE["parameters"])
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM offer_parameters WHERE offer_id='offer-001'").fetchone()[0]
    assert count == 2


# ── writer: images ────────────────────────────────────────────────────────────


def test_upsert_images_inserts_rows():
    conn = mem_db()
    upsert_offer(conn, SAMPLE)
    upsert_images(conn, "offer-001", SAMPLE["images"])
    conn.commit()
    rows = conn.execute(
        "SELECT url, position FROM offer_images WHERE offer_id='offer-001' ORDER BY position"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "https://img.example.com/a.jpg"
    assert rows[0][1] == 0
    assert rows[1][1] == 1


def test_upsert_images_is_idempotent():
    conn = mem_db()
    upsert_offer(conn, SAMPLE)
    upsert_images(conn, "offer-001", SAMPLE["images"])
    upsert_images(conn, "offer-001", SAMPLE["images"])
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM offer_images WHERE offer_id='offer-001'").fetchone()[0] == 2


# ── auth ──────────────────────────────────────────────────────────────────────


def test_get_token_returns_access_token():
    import downloader.auth as auth_mod

    auth_mod._token = None
    auth_mod._token_expires_at = 0.0

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"access_token": "tok_xyz", "expires_in": 3600}
    mock_resp.raise_for_status = MagicMock()

    with patch("downloader.auth.httpx.Client") as MockClient:
        MockClient.return_value.__enter__.return_value.post.return_value = mock_resp
        token = auth_mod.get_token()

    assert token == "tok_xyz"
    auth_mod._token = None
    auth_mod._token_expires_at = 0.0


def test_get_token_uses_cache():
    import downloader.auth as auth_mod

    auth_mod._token = "cached_tok"
    auth_mod._token_expires_at = time.monotonic() + 3600

    with patch("downloader.auth.httpx.Client") as MockClient:
        token = auth_mod.get_token()
        MockClient.assert_not_called()

    assert token == "cached_tok"
    auth_mod._token = None
    auth_mod._token_expires_at = 0.0


# ── client: pagination ────────────────────────────────────────────────────────


def test_iter_all_offers_yields_all_pages(monkeypatch):
    from downloader import client as client_mod

    pages = [
        [{"id": f"o-{i:03d}"} for i in range(100)],
        [{"id": f"o-{i:03d}"} for i in range(100, 150)],
        [],
    ]
    call_idx = {"n": 0}

    def fake_page(offset, limit=100):
        page = pages[call_idx["n"]]
        call_idx["n"] += 1
        return page

    monkeypatch.setattr(client_mod, "_search_page", fake_page)
    offers = list(client_mod.iter_all_offers())
    assert len(offers) == 150


def test_iter_all_offers_respects_max_pages(monkeypatch):
    from downloader import client as client_mod

    def fake_page(offset, limit=100):
        return [{"id": f"o-{offset + i}"} for i in range(100)]

    monkeypatch.setattr(client_mod, "_search_page", fake_page)
    offers = list(client_mod.iter_all_offers(max_pages=2))
    assert len(offers) == 200


def test_iter_all_offers_stops_on_partial_page(monkeypatch):
    from downloader import client as client_mod

    pages = [[{"id": f"o-{i}"} for i in range(42)]]
    call_idx = {"n": 0}

    def fake_page(offset, limit=100):
        page = pages[call_idx["n"]] if call_idx["n"] < len(pages) else []
        call_idx["n"] += 1
        return page

    monkeypatch.setattr(client_mod, "_search_page", fake_page)
    offers = list(client_mod.iter_all_offers())
    assert len(offers) == 42
