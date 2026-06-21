"""Feature vector construction and FAISS similarity index.

`parsed_offers` (built by `scraper/parsed_offers_builder.py`) is the single
source of truth for the tabular feature vector — numeric `spec_*` values are
already validated/typed there, and categorical values are encoded ints
(`*_enc`). This module decodes those ints back to raw strings (via
`category_encoders`) only to bucket them into the same small one-hot
categories used previously; the encoded ints themselves are nominal (encoder
assigns them in encounter order) and would distort distances if fed to
KMeans/FAISS directly as if they were ordinal.

Feature vector layout (all floats, L2-normalised before indexing):
  [0-3]   zero-shot binary attrs × IMAGE_ATTR_WEIGHT
           has_drop_handlebars, has_flat_handlebars, has_disc_brakes, has_pedals
  [4-7]   structured params (one-hot)
           wheel_size (3), frame_material (5), brakes (3), gear_count (1)
  [8]     price (normalised, × PRICE_WEIGHT)

Missing values → 0.0 (FAISS can't handle NaN).
"""
import logging
import sqlite3

import numpy as np

log = logging.getLogger(__name__)

# Lowered from 2.0: at that weight, 4 binary zero-shot dims dominated the
# L2-normalised vector over the 9 structured one-hot dims + price, so "similar
# specs" mostly reflected whichever zero-shot pattern (drop bars/disc brakes/
# etc.) was most common in the dataset rather than material/brakes/price —
# most non-disc-brake bikes looked equally "similar" to each other while the
# rare disc-brake bikes stood out as the only group with real separation.
IMAGE_ATTR_WEIGHT = 0.5
PRICE_WEIGHT = 0.5

# One-hot helpers
_WHEEL_SIZES  = ["700c", "650b", "other"]
_MATERIALS    = ["carbon", "aluminium", "steel", "titanium", "other"]
_BRAKES       = ["disc", "rim", "other"]


def _one_hot(value: str | None, categories: list[str]) -> list[float]:
    low = (value or "").lower()
    return [1.0 if low == c else 0.0 for c in categories]


def _wheel_category(raw_inch: float | None) -> str:
    if raw_inch is None:
        return "other"
    if abs(raw_inch - 28.0) < 0.5:
        return "700c"
    if abs(raw_inch - 27.5) < 0.3:
        return "650b"
    return "other"


def _material_category(raw: str | None) -> str:
    if not raw:
        return "other"
    low = raw.lower()
    for cat in ["carbon", "aluminium", "aluminum", "steel", "titanium"]:
        if cat in low:
            return "aluminium" if cat == "aluminum" else cat
    return "other"


def _brake_category(raw: str | None) -> str:
    if not raw:
        return "other"
    low = raw.lower()
    if "disc" in low or "tarcz" in low:
        return "disc"
    if "rim" in low or "szczęk" in low or "v-brake" in low or "calip" in low:
        return "rim"
    return "other"


def build_feature_vector(offer: dict) -> np.ndarray:
    """Build a fixed-length float32 feature vector for one offer.

    `offer["parsed"]` is a `parsed_offers` row (as a dict); `offer["params"]`
    holds the decoded Materiał ramy/Hamulce strings for one-hot bucketing,
    populated by `_load_offers` below.
    """
    parsed: dict = offer.get("parsed", {})
    params: dict[str, str] = offer.get("params", {})

    # ── Zero-shot binary attrs (weighted) ────────────────────────────────────
    def _zs(key: str) -> float:
        v = parsed.get(key)
        return float(v) if v is not None else 0.0

    image_attrs = np.array([
        _zs("has_drop_handlebars"),
        _zs("has_flat_handlebars"),
        _zs("has_disc_brakes"),
        _zs("has_pedals"),
    ], dtype=np.float32) * IMAGE_ATTR_WEIGHT

    # ── Structured parameters ─────────────────────────────────────────────────
    wheel    = _wheel_category(parsed.get("wheel_size_inch"))
    material = _material_category(params.get("Materiał ramy"))
    brakes   = _brake_category(params.get("Hamulce"))

    try:
        gear_count = float(parsed.get("gear_count") or 0) / 30.0  # rough normalise
    except (TypeError, ValueError):
        gear_count = 0.0

    structured = np.array(
        _one_hot(wheel, _WHEEL_SIZES)
        + _one_hot(material, _MATERIALS)
        + _one_hot(brakes, _BRAKES)
        + [gear_count],
        dtype=np.float32,
    )

    # ── Price (normalised via caller-supplied min/max) ────────────────────────
    price_norm = float(offer.get("price_norm", 0.0)) * PRICE_WEIGHT
    price_feat = np.array([price_norm], dtype=np.float32)

    combined = np.concatenate([image_attrs, structured, price_feat])
    norm = np.linalg.norm(combined)
    if norm > 1e-8:
        combined = combined / norm
    return combined


def _load_decode_maps(conn: sqlite3.Connection) -> dict[str, dict[int, str]]:
    """`{column_name: {encoded_int: raw_value}}` from `category_encoders`."""
    maps: dict[str, dict[int, str]] = {}
    for col, raw, enc in conn.execute(
        "SELECT column_name, raw_value, encoded_int FROM category_encoders"
    ).fetchall():
        maps.setdefault(col, {})[enc] = raw
    return maps


def _load_offers(conn: sqlite3.Connection) -> list[dict]:
    """Load `parsed_offers` rows for bike offers with zero-shot data.

    Decodes `frame_material_enc`/`brake_type_enc` back to raw strings
    (`offer["params"]`) for one-hot bucketing; keeps the full row at
    `offer["parsed"]` for the numeric features `build_feature_vector` reads
    directly.
    """
    decode = _load_decode_maps(conn)
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row
    rows = cur.execute(
        """
        SELECT * FROM parsed_offers
        WHERE id IN (SELECT id FROM offers WHERE is_bike = 1)
          AND has_drop_handlebars IS NOT NULL
        """
    ).fetchall()

    offers = []
    for r in rows:
        parsed = dict(r)
        material_raw = decode.get("frame_material_enc", {}).get(parsed["frame_material_enc"])
        brake_raw = decode.get("brake_type_enc", {}).get(parsed["brake_type_enc"])
        offers.append({
            "id": parsed["id"],
            "price": parsed["price"],
            "parsed": parsed,
            "params": {"Materiał ramy": material_raw, "Hamulce": brake_raw},
        })
    return offers


def _normalise_prices(offers: list[dict]) -> None:
    """Add price_norm (0–1) in-place, ignoring NULL/zero prices."""
    prices = [o["price"] for o in offers if o.get("price")]
    if not prices:
        for o in offers:
            o["price_norm"] = 0.0
        return
    mn, mx = min(prices), max(prices)
    rng = mx - mn or 1.0
    for o in offers:
        p = o.get("price") or mn
        o["price_norm"] = (p - mn) / rng


def _faiss_topk_neighbours(
    conn: sqlite3.Connection,
    ids: list[str],
    vectors: np.ndarray,
    comparison_type: str,
    top_k: int = 5,
) -> int:
    """Build a FAISS IP index over `vectors` and write top-k neighbours per id.

    Vectors must already be L2-normalised so inner product == cosine. Writes
    under the given comparison_type, leaving the other engine's rows intact.
    Also clears any existing rows for offers that dropped out of the current
    candidate set (e.g. reclassified as noise, went inactive) — otherwise
    they'd linger indefinitely since nothing else prunes them.
    """
    try:
        import faiss
    except ImportError:
        log.error("faiss-cpu not installed — run: pip install faiss-cpu")
        return 0

    from scraper.writer import clear_stale_similarity, upsert_similarity

    if len(ids) < 2:
        log.warning("Need ≥2 offers for '%s' similarity — skipping", comparison_type)
        return 0

    cleared = clear_stale_similarity(conn, comparison_type, ids)
    if cleared:
        log.info("Cleared %d stale '%s' similarity row(s)", cleared, comparison_type)

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    k = min(top_k + 1, len(ids))  # first hit is the offer itself
    distances, indices = index.search(vectors, k)

    for i, offer_id in enumerate(ids):
        neighbours = []
        for rank, (j, score) in enumerate(zip(indices[i][1:], distances[i][1:]), start=1):
            if j < 0:
                continue
            neighbours.append({
                "similar_id": ids[j],
                "score": float(score),
                "rank": rank,
            })
        upsert_similarity(conn, offer_id, neighbours, comparison_type=comparison_type)

    conn.commit()
    return len(ids)


def build_similarity_index(conn: sqlite3.Connection) -> int:
    """Specs ('similar specs') engine: top-5 neighbours from the tabular vector."""
    offers = _load_offers(conn)
    if len(offers) < 2:
        log.warning("Need at least 2 offers with zero-shot data to build index")
        return 0

    _normalise_prices(offers)
    vectors = np.stack([build_feature_vector(o) for o in offers]).astype(np.float32)
    n = _faiss_topk_neighbours(conn, [o["id"] for o in offers], vectors, "specs")
    if n:
        try:
            import os
            import faiss
            os.makedirs("db", exist_ok=True)
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(np.ascontiguousarray(vectors))
            faiss.write_index(index, "db/similarity.index")
        except ImportError:
            pass
        log.info("Specs similarity built for %d offers", n)
    return n


def _load_visual_vectors(conn: sqlite3.Connection) -> tuple[list[str], np.ndarray | None]:
    """Load L2-normalised clip_vector_visual embeddings for is_bike=1 offers.

    Returns ([], None) when the visual pipeline (Phase B) hasn't populated any
    embeddings yet — callers skip gracefully.
    """
    import pickle

    rows = conn.execute(
        """
        SELECT o.id, f.clip_vector_visual
        FROM offers o JOIN offer_features f ON f.offer_id = o.id
        WHERE o.is_bike = 1 AND f.clip_vector_visual IS NOT NULL
        """
    ).fetchall()
    if len(rows) < 2:
        return [], None

    ids, vecs = [], []
    for offer_id, blob in rows:
        try:
            v = np.asarray(pickle.loads(blob), dtype=np.float32).reshape(-1)
        except Exception:
            continue
        n = np.linalg.norm(v)
        if n > 1e-8:
            v = v / n
        ids.append(offer_id)
        vecs.append(v)
    if len(ids) < 2:
        return [], None
    return ids, np.stack(vecs).astype(np.float32)


def build_visual_similarity(conn: sqlite3.Connection) -> int:
    """Visual ('similar looking') engine: top-5 neighbours from clip_vector_visual.

    No-op (returns 0) until the visual pipeline has populated clip_vector_visual.
    """
    ids, vectors = _load_visual_vectors(conn)
    if vectors is None:
        log.info("No clip_vector_visual embeddings yet — visual similarity skipped")
        return 0
    n = _faiss_topk_neighbours(conn, ids, vectors, "visual")
    if n:
        log.info("Visual similarity built for %d offers", n)
    return n


def load_vectors_for_clustering(conn: sqlite3.Connection) -> tuple[list[dict], np.ndarray | None]:
    """Return (offers, vectors) for the tabular clustering pass, or ([], None)."""
    offers = _load_offers(conn)
    if not offers:
        return [], None
    _normalise_prices(offers)
    vectors = np.stack([build_feature_vector(o) for o in offers]).astype(np.float32)
    return offers, vectors


def load_visual_for_clustering(
    conn: sqlite3.Connection,
) -> tuple[list[dict], np.ndarray | None]:
    """Return (offers, vectors) for the visual clustering pass, or ([], None).

    offers carry id + price (for cluster avg-price labels); vectors are the
    L2-normalised clip_vector_visual embeddings.
    """
    ids, vectors = _load_visual_vectors(conn)
    if vectors is None:
        return [], None
    price_map = dict(
        conn.execute(
            "SELECT id, price FROM offers WHERE id IN (%s)"
            % ",".join("?" * len(ids)),
            ids,
        ).fetchall()
    )
    offers = [{"id": i, "price": price_map.get(i)} for i in ids]
    return offers, vectors
