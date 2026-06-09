import json
import sqlite3
from datetime import datetime, timezone

_TABLES: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS offers (
        id            TEXT PRIMARY KEY,
        title         TEXT,
        price         REAL,
        currency      TEXT,
        seller_id     TEXT,
        seller_login  TEXT,
        condition     TEXT,
        listing_type  TEXT,
        offer_url     TEXT,
        thumbnail_url TEXT,
        end_time      TEXT,
        fetched_at    TEXT NOT NULL,
        raw_json      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS offer_parameters (
        id        INTEGER PRIMARY KEY,
        offer_id  TEXT NOT NULL,
        name      TEXT NOT NULL,
        value     TEXT,
        UNIQUE (offer_id, name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS offer_images (
        id        INTEGER PRIMARY KEY,
        offer_id  TEXT NOT NULL,
        url       TEXT NOT NULL,
        position  INTEGER NOT NULL DEFAULT 0,
        UNIQUE (offer_id, position)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS offer_features (
        offer_id                 TEXT PRIMARY KEY,
        clip_vector              BLOB,
        description_text         TEXT,
        extracted_groupset       TEXT,
        extracted_frame_material TEXT,
        price_predicted          REAL,
        anomaly_score            REAL
    )
    """,
)


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    for stmt in _TABLES:
        conn.execute(stmt)
    conn.commit()


def _parse_price(offer: dict) -> tuple[float | None, str | None]:
    sm = offer.get("sellingMode", {})
    p = sm.get("price", {})
    try:
        return float(p["amount"]), p.get("currency")
    except (KeyError, TypeError, ValueError):
        return None, p.get("currency")


def _primary_thumbnail(offer: dict) -> str | None:
    imgs = offer.get("images", [])
    return imgs[0].get("url") if imgs else None


def upsert_offer(conn: sqlite3.Connection, raw: dict) -> None:
    offer_id = raw.get("id")
    if not offer_id:
        return
    price, currency = _parse_price(raw)
    seller = raw.get("seller", {})
    conn.execute(
        """
        INSERT INTO offers
            (id, title, price, currency, seller_id, seller_login, condition,
             listing_type, offer_url, thumbnail_url, end_time, fetched_at, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET
            title=excluded.title, price=excluded.price, currency=excluded.currency,
            seller_id=excluded.seller_id, seller_login=excluded.seller_login,
            condition=excluded.condition, listing_type=excluded.listing_type,
            offer_url=excluded.offer_url, thumbnail_url=excluded.thumbnail_url,
            end_time=excluded.end_time, fetched_at=excluded.fetched_at,
            raw_json=excluded.raw_json
        """,
        (
            offer_id,
            raw.get("name") or raw.get("title"),
            price,
            currency,
            seller.get("id"),
            seller.get("login"),
            raw.get("condition"),
            raw.get("sellingMode", {}).get("format"),
            raw.get("url"),
            _primary_thumbnail(raw),
            raw.get("publication", {}).get("endingAt"),
            datetime.now(timezone.utc).isoformat(),
            json.dumps(raw, ensure_ascii=False),
        ),
    )


def upsert_parameters(conn: sqlite3.Connection, offer_id: str, params: list[dict]) -> None:
    for p in params:
        name = p.get("name")
        if not name:
            continue
        values = p.get("values") or p.get("valuesIds") or []
        value_str = ", ".join(str(v) for v in values) if values else p.get("value")
        conn.execute(
            """
            INSERT INTO offer_parameters (offer_id, name, value)
            VALUES (?, ?, ?)
            ON CONFLICT(offer_id, name) DO UPDATE SET value=excluded.value
            """,
            (offer_id, name, value_str),
        )


def upsert_images(conn: sqlite3.Connection, offer_id: str, images: list[dict]) -> None:
    for pos, img in enumerate(images):
        url = img.get("url")
        if url:
            conn.execute(
                """
                INSERT INTO offer_images (offer_id, url, position)
                VALUES (?, ?, ?)
                ON CONFLICT(offer_id, position) DO UPDATE SET url=excluded.url
                """,
                (offer_id, url, pos),
            )
