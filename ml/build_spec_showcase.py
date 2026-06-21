"""Build the spec-extraction showcase (tabular cluster-lab pages).

The tabular analogue of ``ml/build_showcase.py``: instead of showing how an image
becomes a visual embedding, it shows how the **inline Ollama spec extractor**
(``scraper/ollama_extractor.py``, run during the detail scrape) turns a listing's
free-text description into structured ``spec_*`` columns — the same columns that
feed the tabular clustering vector.

For each of a few random offers it captures three stages:

    1. Description           (cleaned via scraper/description_preprocessor.py)
    2. Raw Ollama extraction (whatever the LLM read out of the text)
    3. Validated spec values (after scraper/value_validator.py's unit/range checks —
       a raw value can be corrected, e.g. grams → kg, or dropped as implausible)

Output is ``media/showcase/specs_manifest.json`` (no images — the dashboard reads
it server-side). Re-running reshuffles the selection.

    python ml/build_spec_showcase.py
    python ml/build_spec_showcase.py --count 4 --seed 7

Uses the conda automl env. Needs a running local Ollama (qwen2.5:3b) — offers
whose description yields too few fields are skipped, so an unreachable Ollama
will likely produce an empty manifest.
"""
import argparse
import json
import logging
import random
import sys
from pathlib import Path

from decouple import config

from scraper import spec_columns, value_validator
from scraper.description_preprocessor import preprocess_description
from scraper.ollama_extractor import (
    COLUMN_HUMAN_LABELS,
    build_prompt,
    contexts_for_columns,
    extract_from_description,
)
from scraper.writer import migrate_db, open_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
SHOWCASE_DIR = ROOT / "media" / "showcase"
MANIFEST = SHOWCASE_DIR / "specs_manifest.json"
MIN_FILLED = 3          # only showcase offers rich enough to be illustrative
_LABELS = spec_columns.column_labels()


def _build_prompt(description: str) -> str:
    """Reconstruct the exact prompt `extract_from_description` sends to Ollama —
    system prologue aside, the same few-shot examples, vocabulary, field list and
    full description — so the showcase shows what the model actually sees."""
    return build_prompt(description, contexts_for_columns(list(COLUMN_HUMAN_LABELS)))


def build_spec_entry(offer_id: str, title: str | None, description: str | None) -> dict | None:
    """Clean the description, ask Ollama (verbose schema → evidence + explanation)
    to extract every known spec_* field from it, validate the results, and
    assemble a showcase entry. None if fewer than MIN_FILLED fields came back."""
    cleaned = preprocess_description(description)
    if not cleaned:
        return None

    raw_result = extract_from_description(
        cleaned, list(COLUMN_HUMAN_LABELS), verbose=True
    )
    spec_raw = {k: v for k, v in raw_result.items() if k in COLUMN_HUMAN_LABELS}
    if len(spec_raw) < MIN_FILLED:
        return None

    filled = []
    for col, field in spec_raw.items():
        raw_value = field.get("value")
        fixed, checked = value_validator.validate_and_fix(col, raw_value)
        validated_value = fixed if checked else raw_value
        filled.append({
            "column": col,
            "label": _LABELS.get(col, col),
            "raw_value": raw_value,
            "validated_value": validated_value,
            "evidence": field.get("evidence") or "",
            "explanation": field.get("explanation") or "",
            "source": "ai",
        })

    return {
        "offer_id": offer_id,
        "title": (title or offer_id)[:80],
        "prompt": _build_prompt(cleaned),
        "filled": filled,
    }


def _candidate_rows(conn, count: int, seed: int | None) -> list[tuple]:
    """Random bike offers with a non-empty description — a few extra so we can skip
    any whose text isn't rich enough. A seed makes the pick reproducible."""
    pool = count * 8
    base = (
        "SELECT id, title, description FROM offers "
        "WHERE is_bike = 1 AND description IS NOT NULL AND length(description) > 0"
    )
    if seed is not None:
        rows = conn.execute(base + " ORDER BY id").fetchall()
        random.seed(seed)
        return random.sample(rows, min(pool, len(rows)))
    return conn.execute(base + " ORDER BY RANDOM() LIMIT ?", (pool,)).fetchall()


def build_spec_showcase(conn, count: int = 3, seed: int | None = None) -> list[dict]:
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for offer_id, title, description in _candidate_rows(conn, count, seed):
        if len(manifest) >= count:
            break
        entry = build_spec_entry(offer_id, title, description)
        if entry is None:
            continue
        manifest.append(entry)
        log.info("spec showcase for %s (%d columns)", offer_id, len(entry["filled"]))

    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the spec-extraction showcase")
    parser.add_argument("--count", type=int, default=3, help="number of offers to showcase")
    parser.add_argument("--seed", type=int, default=None, help="reproducible selection")
    args = parser.parse_args()

    conn = open_db(config("DB_PATH"))
    migrate_db(conn)
    manifest = build_spec_showcase(conn, count=args.count, seed=args.seed)
    conn.close()
    log.info("Spec showcase built: %d offers → %s", len(manifest), MANIFEST)


if __name__ == "__main__":
    main()
